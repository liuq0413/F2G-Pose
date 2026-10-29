"""Run Meta SAM 3 in its own Python environment and save instance masks."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image


# 直接执行本文件时，Python 会把 ``src/f2g_pose`` 放在 sys.path 首位。
# 该目录也有 F2G-Pose 的 ``sam3.py`` 包装模块，会遮蔽独立环境里的
# Meta SAM 3 ``sam3`` 包；移除脚本目录后再导入官方包。
_worker_dir = str(Path(__file__).resolve().parent)
sys.path = [entry for entry in sys.path
            if str(Path(entry or '.').resolve()) != _worker_dir]


def save_masks(output, masks, scores, shape, prompt):
    output = Path(output)
    masks = np.asarray(masks)
    scores = np.asarray(scores).reshape(-1)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3 or masks.shape[1:] != shape or len(masks) != len(scores):
        raise ValueError('SAM 3 masks must have shape N×1×H×W or N×H×W and match RGB.')
    # 按图像中的水平位置排序，使多目标文件名稳定且便于逐个推理。
    instances = []
    for mask, score in zip(masks, scores):
        ys, xs = np.where(mask > 0)
        if len(xs):
            instances.append((float(xs.mean()), float(ys.mean()), mask, float(score)))
    instances.sort(key=lambda item: (item[0], item[1]))
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for index, (_, _, mask, score) in enumerate(instances, 1):
        name = f'mask_{index:02d}.png'
        Image.fromarray((mask > 0).astype(np.uint8) * 255).save(output / name)
        records.append({'file': name, 'score': score, 'pixels': int((mask > 0).sum())})
    (output / 'masks.json').write_text(json.dumps({'prompt': prompt, 'instances': records}, indent=2) + '\n')
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description='Generate one binary mask per SAM 3 instance.')
    parser.add_argument('--rgb', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--prompt', default='water bottle')
    parser.add_argument('--confidence', type=float, default=0.5)
    parser.add_argument('--checkpoint', help='Local SAM 3 checkpoint; otherwise use authenticated Hugging Face cache')
    args = parser.parse_args(argv)
    if not args.prompt.strip() or not 0 <= args.confidence <= 1:
        parser.error('Prompt must be nonempty and confidence must be between 0 and 1.')

    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    image = Image.open(args.rgb).convert('RGB')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_sam3_image_model(device=device, checkpoint_path=args.checkpoint,
                                   load_from_HF=args.checkpoint is None)
    processor = Sam3Processor(model, device=device, confidence_threshold=args.confidence)
    # SAM 3 的 ViT 特征路径在 CUDA 上使用 BF16；autocast 同时处理
    # 浮点权重和 BF16 激活，避免 Linear 出现 dtype mismatch。
    amp = (torch.autocast(device_type='cuda', dtype=torch.bfloat16)
           if device == 'cuda' else nullcontext())
    with torch.inference_mode(), amp:
        state = processor.set_image(image)
        result = processor.set_text_prompt(state=state, prompt=args.prompt)
    records = save_masks(args.output, result['masks'].cpu().numpy(),
                         result['scores'].float().cpu().numpy(),
                         (image.height, image.width), args.prompt)
    print(f'Saved {len(records)} mask(s) to {args.output}')


if __name__ == '__main__':
    main()
