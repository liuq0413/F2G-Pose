#!/usr/bin/env python3
"""Download the gated facebook/sam3 repository to a local directory.

Authentication is read from ``HF_TOKEN`` when it is set.  If it is not set,
``huggingface_hub`` uses the token saved by ``hf auth login``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("weights/sam3"),
        help="本地模型目录（默认：weights/sam3）",
    )
    parser.add_argument(
        "--repo-id",
        default="facebook/sam3",
        help="Hugging Face 模型仓库（默认：facebook/sam3）",
    )
    parser.add_argument(
        "--revision",
        help="可选的模型 revision、branch 或 commit",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "缺少 huggingface_hub。请在 SAM3 环境执行："
            "python -m pip install -U huggingface_hub"
        ) from exc

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    # 优先使用环境变量，避免把令牌写入 shell 历史；未设置时沿用 hf auth login。
    token = os.environ.get("HF_TOKEN") or None
    print(f"Downloading {args.repo_id} to {output}")
    downloaded = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=str(output),
        token=token,
    )
    print(f"SAM3 files are available at {Path(downloaded).resolve()}")
    print("若使用 Hugging Face Transformers，可将该目录传给 from_pretrained。")


if __name__ == "__main__":
    main()
