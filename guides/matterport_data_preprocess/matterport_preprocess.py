"""Matterport3D preprocessor for PanoVGGT.

Produces, under <out_root>/:
    parsed_json/<scene>.json       room_id -> {room_name, panoramas}
    <scene>/pano_skybox_color/<pano>.png   1024x2048 equirectangular RGB
    <scene>/pano_depth/<pano>.png          1024x2048 uint16 (depth_m * 4000)
    <scene>/pano_poses/<pano>.txt          4x4 c2w in Matterport (OpenGL) frame
    train.txt / val.txt / test.txt         scene id per line, from PanoVGGT splits

Run on one scene:
    python guides/matterport_data_preprocess/matterport_preprocess.py --scene 17DRP5sb8fy
Run on all scenes:
    python guides/matterport_data_preprocess/matterport_preprocess.py --all -j 8
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import os.path as osp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("mp3d_preprocess")

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
DEFAULT_SCANS_DIR = osp.abspath(osp.join(REPO_ROOT, '..', 'dataset', 'Matterport3D', 'v1', 'scans'))
DEFAULT_OUT_ROOT = osp.abspath(osp.join(REPO_ROOT, '..', 'dataset', 'Matterport3D', 'processed'))
DEFAULT_SPLITS_DIR = osp.join(REPO_ROOT, 'training', 'data', 'splits', 'matterport3d')

ERP_H = 1024
ERP_W = 2048
DEPTH_SCALE = 4000.0  # meters -> uint16
DEPTH_MAX_M = 16.0    # truncate beyond this; uint16 max ~65535 -> safe at 16m * 4000 = 64000
SKYBOX_FACE = 1024    # input skybox face resolution

# Skybox face -> rotation that maps a face-camera ray (OpenCV cam frame:
# x-right, y-down, z-forward) into the panorama frame (OpenGL: x-right, y-up,
# z-back). The Matterport skybox order was verified by raw face boundary
# continuity: 0 top, 1 front, 2 right, 3 back, 4 left, 5 down. The common bug
# is to swap the right/left interpretation of faces 2 and 4, which creates
# visibly misaligned equirectangular panoramas.
SKYBOX_R_PANO_FROM_FACE = np.array([
    [[ 1,  0,  0], [ 0,  0,  1], [ 0, -1,  0]],  # top (+y)
    [[ 1,  0,  0], [ 0, -1,  0], [ 0,  0, -1]],  # front (-z)
    [[ 0,  0,  1], [ 0, -1,  0], [ 1,  0,  0]],  # right (+x)
    [[-1,  0,  0], [ 0, -1,  0], [ 0,  0,  1]],  # back (+z)
    [[ 0,  0, -1], [ 0, -1,  0], [-1,  0,  0]],  # left (-x)
    [[ 1,  0,  0], [ 0,  0, -1], [ 0,  1,  0]],  # down (-y)
], dtype=np.float64)


@dataclass
class HouseInfo:
    pano_pos: dict[str, np.ndarray]                    # pano -> (3,) world position
    pano_to_region: dict[str, int]                     # pano -> region id
    region_label: dict[int, str]                       # region id -> letter label


def parse_house(scene: str, scan_dir: str) -> HouseInfo:
    """Parse <scene>.house and panorama_to_region.txt.

    .house format docs: niessner/Matterport/data_organization.md.
    Lines we care about:
        P  <pano_id>  <pano_idx> <region_idx> 0  px py pz  ...
        R  <r_idx> <level_idx> 0 0  <label_letter> ...

    Note on the I (image) lines: they DO contain a 4x4 matrix, but it is the
    *world-to-camera* extrinsic in **OpenGL** camera convention (y-up, z-back).
    The per-image .txt files in matterport_camera_poses/ are the camera-to-
    world version in **OpenCV** convention. Always read poses from the .txt
    files (via _read_pose_txt) instead of the I lines, since the conventions
    are easy to confuse and the .txt is the canonical source.
    """
    seg_dir = osp.join(scan_dir, scene, "house_segmentations")
    house_file = osp.join(seg_dir, f"{scene}.house")
    p2r_file = osp.join(seg_dir, "panorama_to_region.txt")
    if not osp.isfile(house_file):
        raise FileNotFoundError(house_file)

    pano_pos: dict[str, np.ndarray] = {}
    pano_to_region: dict[str, int] = {}
    region_label: dict[int, str] = {}

    if osp.isfile(p2r_file):
        with open(p2r_file) as fh:
            for line in fh:
                t = line.split()
                if len(t) >= 3:
                    pano_to_region[t[1]] = int(t[2])

    with open(house_file) as fh:
        for line in fh:
            t = line.split()
            if not t:
                continue
            tag = t[0]
            if tag == "R":
                rid = int(t[1])
                if len(t) >= 6:
                    region_label[rid] = t[5]
            elif tag == "P":
                pano_id = t[1]
                rid = int(t[3])
                px, py, pz = float(t[5]), float(t[6]), float(t[7])
                pano_pos[pano_id] = np.array([px, py, pz], dtype=np.float64)
                pano_to_region.setdefault(pano_id, rid)

    return HouseInfo(pano_pos, pano_to_region, region_label)


def write_parsed_json(info: HouseInfo, scene: str, out_root: str) -> None:
    """Group panoramas by region and emit parsed_json/<scene>.json."""
    rooms: dict[int, dict] = {}
    for pano, rid in info.pano_to_region.items():
        if rid < 0:
            continue  # unassigned panorama
        room = rooms.setdefault(rid, {"room_name": info.region_label.get(rid, f"room_{rid}"),
                                       "panoramas": []})
        room["panoramas"].append(pano)
    for rid in rooms:
        rooms[rid]["panoramas"].sort()

    parsed_dir = osp.join(out_root, "parsed_json")
    os.makedirs(parsed_dir, exist_ok=True)
    out = {str(rid): rooms[rid] for rid in sorted(rooms)}
    with open(osp.join(parsed_dir, f"{scene}.json"), "w") as fh:
        json.dump(out, fh, indent=2)


# ---- skybox -> equirectangular RGB ---------------------------------------

def _build_skybox_remap(equ_h: int, equ_w: int, face_size: int):
    """Precompute per-face (map_x, map_y, mask) for cv2.remap.

    For each ERP pixel, decide which of the 6 faces its ray hits and where on
    that face. The face frame matches SKYBOX_R_PANO_FROM_FACE: a ray d_pano in
    the pano frame is converted to the face frame by R_face_from_pano = R^T.
    The face camera has principal axis +z (i.e. -z = look direction in the
    panovggt convention). For the cubemap unit cube, the face is z=+1
    (cv2.remap reads pixels at (u,v) so we use a pinhole with f = face/2,
    cx = cy = (face-1)/2).
    """
    # ERP ray directions in the pano frame.
    # Convention for the output ERP: longitude theta runs from -pi (left) to +pi (right),
    # latitude phi from -pi/2 (bottom) to +pi/2 (top). Theta=0 looks along the
    # pano's -z axis (skybox-1 forward), so: dir = (sin th cos ph, sin ph, -cos th cos ph).
    # That puts the front face at the center of the ERP, which matches PanoBasic.
    th = (np.arange(equ_w) + 0.5) / equ_w  # [0,1)
    th = (th - 0.5) * 2 * np.pi            # [-pi, pi)
    ph = (np.arange(equ_h) + 0.5) / equ_h
    ph = (0.5 - ph) * np.pi                # [+pi/2, -pi/2)
    th, ph = np.meshgrid(th, ph)
    cph = np.cos(ph)
    dx = np.sin(th) * cph
    dy = np.sin(ph)
    dz = -np.cos(th) * cph
    rays_pano = np.stack([dx, dy, dz], axis=-1)  # (H,W,3)

    # Decide face for each ERP pixel: face f is selected when its forward axis
    # aligns most with the ray. Forward in the pano frame for face f is
    # SKYBOX_R_PANO_FROM_FACE[f] @ (0,0,1).
    fwd_pano = SKYBOX_R_PANO_FROM_FACE @ np.array([0, 0, 1], dtype=np.float64)  # (6,3)
    dots = rays_pano @ fwd_pano.T  # (H,W,6)
    face_idx = np.argmax(dots, axis=-1)

    f = face_size / 2.0
    cxcy = (face_size - 1) / 2.0
    maps = []
    for fi in range(6):
        R_face_from_pano = SKYBOX_R_PANO_FROM_FACE[fi].T
        ray_face = rays_pano @ R_face_from_pano.T  # (H,W,3) — OpenCV cam frame (z-fwd, y-down)
        z = ray_face[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = f * ray_face[..., 0] / z + cxcy
            v = f * ray_face[..., 1] / z + cxcy
        mask = (face_idx == fi) & (z > 0)
        u = np.where(mask, u, 0.0).astype(np.float32)
        v = np.where(mask, v, 0.0).astype(np.float32)
        maps.append((u, v, mask))
    return maps


_REMAP_CACHE: dict[tuple[int, int, int], list] = {}


def stitch_skybox_rgb(scene_dir: str, pano: str, equ_h: int = ERP_H, equ_w: int = ERP_W,
                      face_size: int = SKYBOX_FACE) -> np.ndarray:
    key = (equ_h, equ_w, face_size)
    maps = _REMAP_CACHE.get(key)
    if maps is None:
        maps = _build_skybox_remap(equ_h, equ_w, face_size)
        _REMAP_CACHE[key] = maps

    out = np.zeros((equ_h, equ_w, 3), dtype=np.uint8)
    sb_dir = osp.join(scene_dir, "matterport_skybox_images")
    for fi in range(6):
        path = osp.join(sb_dir, f"{pano}_skybox{fi}_sami.jpg")
        face = cv2.imread(path, cv2.IMREAD_COLOR)
        if face is None:
            raise FileNotFoundError(path)
        if face.shape[0] != face_size:
            face = cv2.resize(face, (face_size, face_size), interpolation=cv2.INTER_AREA)
        u, v, mask = maps[fi]
        warped = cv2.remap(face, u, v, interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
        out[mask] = warped[mask]
    return out  # BGR


# ---- 18-view perspective depth -> equirectangular depth -------------------

def _read_intrinsics(scene_dir: str, pano: str, cam_idx: int) -> tuple[np.ndarray, int, int]:
    p = osp.join(scene_dir, "matterport_camera_intrinsics", f"{pano}_intrinsics_{cam_idx}.txt")
    t = np.fromstring(open(p).read(), sep=" ", dtype=np.float64)
    # Format: W H fx fy cx cy k1 k2 p1 p2 k3
    W, H = int(t[0]), int(t[1])
    K = np.array([[t[2], 0, t[4]], [0, t[3], t[5]], [0, 0, 1]], dtype=np.float64)
    return K, W, H


def _read_pose_txt(scene_dir: str, pano: str, cam_idx: int, yaw_idx: int) -> np.ndarray:
    p = osp.join(scene_dir, "matterport_camera_poses",
                 f"{pano}_pose_{cam_idx}_{yaw_idx}.txt")
    return np.loadtxt(p, dtype=np.float64)


def stitch_depth(scene_dir: str, pano: str, pano_c2w_gl: np.ndarray,
                 equ_h: int = ERP_H, equ_w: int = ERP_W) -> np.ndarray:
    """Reproject 18 perspective depth views into ERP, min-radius blended.

    pano_c2w_gl is the pano c2w in MP3D-world, **OpenGL-cam** (y-up, z-back) —
    same frame the ERP RGB was rendered for, where theta=0 looks along -z.
    """
    R_pano_w = pano_c2w_gl[:3, :3]

    erp_r = np.full((equ_h, equ_w), np.inf, dtype=np.float32)
    written = 0

    for cam_idx in range(3):
        K, W, H = _read_intrinsics(scene_dir, pano, cam_idx)
        K_inv = np.linalg.inv(K)
        # Build pixel grid once per camera.
        uu, vv = np.meshgrid(np.arange(W), np.arange(H))
        pix = np.stack([uu.ravel(), vv.ravel(), np.ones(W * H)], axis=0)  # (3, N)
        cam_rays = K_inv @ pix  # (3, N) directions in cam frame (z=1 plane)

        for yaw_idx in range(6):
            dpath = osp.join(scene_dir, "matterport_depth_images",
                             f"{pano}_d{cam_idx}_{yaw_idx}.png")
            d = cv2.imread(dpath, cv2.IMREAD_UNCHANGED)
            if d is None:
                continue
            if d.shape != (H, W):
                d = cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST)
            z = d.astype(np.float32) / 4000.0  # Matterport stores depth in 0.25mm units
            valid = (d > 0) & (z < DEPTH_MAX_M)
            if not valid.any():
                continue

            T_view = _read_pose_txt(scene_dir, pano, cam_idx, yaw_idx)
            R_view_w = T_view[:3, :3]  # OpenCV cam in MP3D world
            # cam_rays are OpenCV-cam frame (x=right, y=down, z=forward).
            # Apply R_view_w to lift to world, then R_pano_w^T to drop into the
            # pano (OpenGL) frame. R_pano_w was derived from a perspective view
            # via R_pano_w = R_view_pose1_w @ diag(1,-1,-1), so this works out:
            # the ratio of pano-frame to world-frame is OpenGL on the cam side.
            world_rays = R_view_w @ cam_rays                            # (3, N)
            pano_rays = R_pano_w.T @ world_rays                          # (3, N) OpenGL frame

            # Convert to (theta, phi, r) using the same convention as the RGB
            # remap above: theta=atan2(x, -z), phi=asin(y/r), r=||xyz||.
            # Distance "r" we want is z (perspective depth) projected to a
            # *radial* distance to the pano center. Per ray, point = z * cam_ray
            # in cam coords; ||point|| = z * ||cam_ray||. Since rotation
            # preserves length, r = z * ||cam_ray|| (norm of the (3,N) vector).
            ray_norm = np.linalg.norm(cam_rays, axis=0)  # (N,)
            r_flat = z.ravel() * ray_norm

            # pano_rays is the *direction* (pre-norm). Normalize for theta/phi.
            pr = pano_rays / np.maximum(np.linalg.norm(pano_rays, axis=0, keepdims=True), 1e-9)
            theta = np.arctan2(pr[0], -pr[2])             # [-pi, pi]
            phi = np.arcsin(np.clip(pr[1], -1.0, 1.0))     # [-pi/2, pi/2]

            u_erp = ((theta / (2 * np.pi)) + 0.5) * equ_w - 0.5
            v_erp = (0.5 - phi / np.pi) * equ_h - 0.5
            u_erp = np.clip(np.round(u_erp).astype(np.int32), 0, equ_w - 1)
            v_erp = np.clip(np.round(v_erp).astype(np.int32), 0, equ_h - 1)

            valid_flat = valid.ravel()
            u_erp = u_erp[valid_flat]
            v_erp = v_erp[valid_flat]
            r_flat = r_flat[valid_flat]

            # Unbuffered min reduction: np.minimum.at handles duplicate indices
            # correctly, unlike vanilla fancy-index assignment.
            flat = v_erp.astype(np.int64) * equ_w + u_erp.astype(np.int64)
            np.minimum.at(erp_r.ravel(), flat, r_flat)
            written += int(r_flat.size)

    erp_r[~np.isfinite(erp_r)] = 0.0
    return erp_r  # in meters


# ---- per-scene driver -----------------------------------------------------

def process_pano(scene_dir: str, pano: str, info: HouseInfo, scene_out_dir: str) -> None:
    color_dir = osp.join(scene_out_dir, "pano_skybox_color")
    depth_dir = osp.join(scene_out_dir, "pano_depth")
    pose_dir = osp.join(scene_out_dir, "pano_poses")
    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(pose_dir, exist_ok=True)

    color_path = osp.join(color_dir, f"{pano}.png")
    depth_path = osp.join(depth_dir, f"{pano}.png")
    pose_path = osp.join(pose_dir, f"{pano}.txt")

    if osp.exists(color_path) and osp.exists(depth_path) and osp.exists(pose_path):
        return

    # Pano c2w = camera 1, yaw 5's c2w. The .txt is c2w in MP3D-world (Z-up)
    # with OpenCV camera (y-down, z-forward). Downstream loader expects
    # OpenGL-cam (y-up, z-back), so flip the camera Y/Z columns. Same flip
    # aligns the pose with the pano frame used by ERP RGB / depth stitchers.
    txt_path = osp.join(scene_dir, "matterport_camera_poses",
                        f"{pano}_pose_1_5.txt")
    if not osp.isfile(txt_path):
        log.warning("missing pose .txt for %s cam=1 yaw=5", pano)
        return
    raw_c2w = np.loadtxt(txt_path)
    cam_flip = np.diag([1.0, -1.0, -1.0, 1.0])
    pano_c2w_gl = (raw_c2w @ cam_flip).astype(np.float32)

    rgb = stitch_skybox_rgb(scene_dir, pano)
    cv2.imwrite(color_path, rgb, [cv2.IMWRITE_PNG_COMPRESSION, 3])

    depth_m = stitch_depth(scene_dir, pano, pano_c2w_gl)
    depth_u16 = np.clip(depth_m * DEPTH_SCALE, 0, 65535).astype(np.uint16)
    cv2.imwrite(depth_path, depth_u16)

    np.savetxt(pose_path, pano_c2w_gl, fmt="%.8f")


def process_scene(scene: str, scans_dir: str, out_root: str) -> str:
    scene_dir = osp.join(scans_dir, scene)
    scene_out = osp.join(out_root, scene)
    log.info("scene %s start", scene)
    info = parse_house(scene, scans_dir)
    write_parsed_json(info, scene, out_root)

    panos = sorted(info.pano_pos.keys())
    for pano in panos:
        try:
            process_pano(scene_dir, pano, info, scene_out)
        except Exception as e:
            log.exception("pano %s/%s failed: %s", scene, pano, e)
    log.info("scene %s done (%d panos)", scene, len(panos))
    return scene


def write_split_files(out_root: str, splits_dir: str) -> None:
    """Translate PanoVGGT split JSONs (lists of [scene, room_id, ...]) into
    per-line scene id files (val.txt etc.). The loader reads {mode}.txt where
    mode is 'train' / 'val' / 'test'."""
    for split in ("train", "val", "test"):
        path = osp.join(splits_dir, f"matterport3d_{split}_index.json")
        with open(path) as fh:
            entries = json.load(fh)
        scenes = sorted({entry[0] for entry in entries})
        with open(osp.join(out_root, f"{split}.txt"), "w") as fh:
            fh.write("\n".join(scenes) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scans-dir", default=DEFAULT_SCANS_DIR)
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--splits-dir", default=DEFAULT_SPLITS_DIR)
    ap.add_argument("--scene", action="append", help="process specific scene id (repeatable)")
    ap.add_argument("--all", action="store_true", help="process all 90 scenes")
    ap.add_argument("-j", "--workers", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    if args.all:
        scenes = sorted(d for d in os.listdir(args.scans_dir)
                        if osp.isdir(osp.join(args.scans_dir, d)))
    else:
        scenes = args.scene or []
    if not scenes:
        ap.error("specify --scene <id> or --all")

    if args.workers <= 1:
        for s in scenes:
            process_scene(s, args.scans_dir, args.out_root)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_scene, s, args.scans_dir, args.out_root): s
                    for s in scenes}
            for fut in as_completed(futs):
                fut.result()

    write_split_files(args.out_root, args.splits_dir)
    log.info("split files written to %s", args.out_root)


if __name__ == "__main__":
    main()
