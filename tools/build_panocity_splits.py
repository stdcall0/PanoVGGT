import json
import os
import sys


def build_splits(root: str, chunk_size: int = 24) -> dict:
    splits = []

    for city in sorted(os.listdir(root)):
        city_path = os.path.join(root, city)
        if not os.path.isdir(city_path) or city.startswith("."):
            continue

        for block in sorted(os.listdir(city_path)):
            block_path = os.path.join(city_path, block)
            if not os.path.isdir(block_path):
                continue

            block_suffix = block[len(f"{city}_") :] if block.startswith(f"{city}_") else block
            poses_file = os.path.join(block_path, f"{city}_Pano_{block_suffix}_poses.json")
            pano_dir = os.path.join(block_path, "pano_images")
            depth_dir = os.path.join(block_path, "panodepth_images")

            if not (os.path.isfile(poses_file) and os.path.isdir(pano_dir) and os.path.isdir(depth_dir)):
                continue

            with open(poses_file, "r") as f:
                pose_payload = json.load(f)

            valid_pairs = []
            for frame in pose_payload.get("frames", []):
                rgb_name = frame.get("name")
                depth_name = frame.get("depth")
                if not rgb_name or not depth_name:
                    continue

                rgb_path = os.path.join(pano_dir, rgb_name)
                depth_path = os.path.join(depth_dir, depth_name)
                if os.path.isfile(rgb_path) and os.path.isfile(depth_path):
                    valid_pairs.append((rgb_name, depth_name))

            if len(valid_pairs) < chunk_size:
                continue

            rel_pose = os.path.relpath(poses_file, root)
            rel_pano = [
                os.path.relpath(os.path.join(pano_dir, rgb_name), root)
                for rgb_name, _ in valid_pairs
            ]
            rel_depth = [
                os.path.relpath(os.path.join(depth_dir, depth_name), root)
                for _, depth_name in valid_pairs
            ]

            for start in range(0, len(valid_pairs) - chunk_size + 1, chunk_size):
                end = start + chunk_size
                splits.append(
                    {
                        "scene": city,
                        "block": block,
                        "part_id": len(splits),
                        "pano_images": rel_pano[start:end],
                        "panodepth_images": rel_depth[start:end],
                        "poses_file": rel_pose,
                        "indices": list(range(start, end)),
                    }
                )

    return {"total_splits": len(splits), "splits": splits}


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python tools/build_panocity_splits.py <panocity_root> [chunk_size]")
        return 1

    root = os.path.abspath(sys.argv[1])
    chunk_size = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    output_path = os.path.join(root, "splits_config.json")

    payload = build_splits(root, chunk_size=chunk_size)
    with open(output_path, "w") as f:
        json.dump(payload, f)

    print(output_path)
    print(f"total_splits={payload['total_splits']}")
    if payload["splits"]:
        first = payload["splits"][0]
        print(
            f"first={first['scene']} {first['block']} "
            f"frames={len(first['pano_images'])} poses={first['poses_file']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
