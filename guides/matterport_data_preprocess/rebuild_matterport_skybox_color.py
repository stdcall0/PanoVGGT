#!/usr/bin/env python3
"""Rebuild Matterport3D skybox RGB panoramas with corrected face geometry.

The original processed Matterport RGB panoramas in this workspace were built
with skybox faces 2 and 4 interpreted as left/right in the wrong orientation.
This script rebuilds only the RGB equirectangular panorama from the six raw
`matterport_skybox_images/*_skybox{0..5}_sami.jpg` faces. It does not touch
processed depth or poses.
"""
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from typing import Iterable

import cv2
import numpy as np

DEFAULT_SCANS_DIR = "../dataset/Matterport3D/v1/scans"
DEFAULT_PROCESSED_ROOT = "../dataset/Matterport3D/processed"
ERP_H = 1024
ERP_W = 2048
FACE_SIZE = 1024

# Face-camera OpenCV axes (x-right, y-down, z-forward) into the pano OpenGL
# frame (x-right, y-up, z-back). ERP theta=0 looks along pano -z.
# Matterport skybox face order, verified by raw face boundary continuity:
#   0 top(+y), 1 front(-z), 2 right(+x), 3 back(+z), 4 left(-x), 5 down(-y)
SKYBOX_R_PANO_FROM_FACE = np.array([
    [[ 1,  0,  0], [ 0,  0,  1], [ 0, -1,  0]],  # top
    [[ 1,  0,  0], [ 0, -1,  0], [ 0,  0, -1]],  # front
    [[ 0,  0,  1], [ 0, -1,  0], [ 1,  0,  0]],  # right
    [[-1,  0,  0], [ 0, -1,  0], [ 0,  0,  1]],  # back
    [[ 0,  0, -1], [ 0, -1,  0], [-1,  0,  0]],  # left
    [[ 1,  0,  0], [ 0,  0, -1], [ 0,  1,  0]],  # down
], dtype=np.float64)


def _load_pano_ids(scene: str, processed_root: str, scans_dir: str) -> list[str]:
    parsed = osp.join(processed_root, "parsed_json", f"{scene}.json")
    if osp.isfile(parsed):
        with open(parsed, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        ids = sorted({p for room in data.values() for p in room.get("panoramas", [])})
        if ids:
            return ids

    sky_dir = osp.join(scans_dir, scene, "matterport_skybox_images")
    ids = set()
    if osp.isdir(sky_dir):
        for name in os.listdir(sky_dir):
            marker = "_skybox"
            if name.endswith("_sami.jpg") and marker in name:
                ids.add(name.split(marker, 1)[0])
    return sorted(ids)


@lru_cache(maxsize=8)
def build_skybox_remap(equ_h: int, equ_w: int, face_size: int):
    th = (np.arange(equ_w, dtype=np.float64) + 0.5) / equ_w
    th = (th - 0.5) * 2.0 * np.pi
    ph = (np.arange(equ_h, dtype=np.float64) + 0.5) / equ_h
    ph = (0.5 - ph) * np.pi
    th, ph = np.meshgrid(th, ph)
    cph = np.cos(ph)
    rays_pano = np.stack([
        np.sin(th) * cph,
        np.sin(ph),
        -np.cos(th) * cph,
    ], axis=-1)

    fwd_pano = SKYBOX_R_PANO_FROM_FACE @ np.array([0.0, 0.0, 1.0])
    face_idx = np.argmax(rays_pano @ fwd_pano.T, axis=-1)

    f = face_size / 2.0
    c = (face_size - 1) / 2.0
    maps = []
    for face_id in range(6):
        R_face_from_pano = SKYBOX_R_PANO_FROM_FACE[face_id].T
        ray_face = rays_pano @ R_face_from_pano.T
        z = ray_face[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            map_x = f * ray_face[..., 0] / z + c
            map_y = f * ray_face[..., 1] / z + c
        mask = (face_idx == face_id) & (z > 0)
        maps.append((map_x.astype(np.float32), map_y.astype(np.float32), mask))
    return maps


def stitch_skybox_rgb(scene_dir: str, pano_id: str, equ_h: int, equ_w: int, face_size: int) -> np.ndarray:
    maps = build_skybox_remap(equ_h, equ_w, face_size)
    sky_dir = osp.join(scene_dir, "matterport_skybox_images")
    out = np.zeros((equ_h, equ_w, 3), dtype=np.uint8)
    for face_id, (map_x, map_y, mask) in enumerate(maps):
        path = osp.join(sky_dir, f"{pano_id}_skybox{face_id}_sami.jpg")
        face = cv2.imread(path, cv2.IMREAD_COLOR)
        if face is None:
            raise FileNotFoundError(path)
        if face.shape[:2] != (face_size, face_size):
            face = cv2.resize(face, (face_size, face_size), interpolation=cv2.INTER_AREA)
        warped = cv2.remap(
            face,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        out[mask] = warped[mask]
    return out


def process_one(args_tuple):
    scans_dir, processed_root, scene, pano_id, out_subdir, equ_h, equ_w, face_size, overwrite = args_tuple
    scene_dir = osp.join(scans_dir, scene)
    out_dir = osp.join(processed_root, scene, out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = osp.join(out_dir, f"{pano_id}.png")
    if osp.exists(out_path) and not overwrite:
        return out_path, "skip"
    rgb = stitch_skybox_rgb(scene_dir, pano_id, equ_h, equ_w, face_size)
    ok = cv2.imwrite(out_path, rgb, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise IOError(f"cv2.imwrite failed: {out_path}")
    return out_path, "write"


def expand_jobs(args) -> list[tuple]:
    scenes = args.scene or []
    if args.all:
        scenes = sorted(
            d for d in os.listdir(args.scans_dir)
            if osp.isdir(osp.join(args.scans_dir, d))
        )
    if not scenes:
        raise SystemExit("specify --scene <id> or --all")

    requested = set(args.pano or [])
    jobs = []
    for scene in scenes:
        pano_ids = sorted(requested) if requested else _load_pano_ids(scene, args.processed_root, args.scans_dir)
        for pano_id in pano_ids:
            jobs.append((
                args.scans_dir,
                args.processed_root,
                scene,
                pano_id,
                args.out_subdir,
                args.equ_h,
                args.equ_w,
                args.face_size,
                args.overwrite,
            ))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scans-dir", default=DEFAULT_SCANS_DIR)
    parser.add_argument("--processed-root", default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--scene", action="append")
    parser.add_argument("--pano", action="append", help="specific panorama id; repeatable")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--out-subdir", default="pano_skybox_color_fixed")
    parser.add_argument("--equ-h", type=int, default=ERP_H)
    parser.add_argument("--equ-w", type=int, default=ERP_W)
    parser.add_argument("--face-size", type=int, default=FACE_SIZE)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("-j", "--workers", type=int, default=1)
    args = parser.parse_args()

    jobs = expand_jobs(args)
    print(f"jobs: {len(jobs)}")
    written = skipped = 0
    if args.workers <= 1:
        for job in jobs:
            path, status = process_one(job)
            written += status == "write"
            skipped += status == "skip"
            print(f"{status}: {path}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_one, job) for job in jobs]
            for future in as_completed(futures):
                path, status = future.result()
                written += status == "write"
                skipped += status == "skip"
                print(f"{status}: {path}")
    print(f"done: written={written} skipped={skipped}")


if __name__ == "__main__":
    main()
