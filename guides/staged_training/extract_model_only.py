#!/usr/bin/env python3
"""Extract model weights from a trainer checkpoint for the next training stage."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='source trainer checkpoint, e.g. checkpoint.pt')
    parser.add_argument('--output', required=True, help='output model-only checkpoint')
    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    ckpt = torch.load(src, map_location='cpu', weights_only=False)
    model = ckpt.get('model', ckpt)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model}, dst)
    print(f'wrote {dst} with {len(model)} tensors')


if __name__ == '__main__':
    main()
