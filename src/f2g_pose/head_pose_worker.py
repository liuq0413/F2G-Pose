"""按需分割头部相机目标，并连续估计新采集帧中的相机系物体位姿。"""
import argparse
import json
from pathlib import Path
from collections import deque
from queue import Empty, Queue
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from .estimator import PoseEstimator
from .head_camera import fetch_frame
from .sam3 import generate_masks
from .sam3 import WORKER as SAM3_WORKER


class Sam3Client:
    """通过独立 Python 进程复用已加载的 SAM 3 模型。"""

    def __init__(self, python, checkpoint, confidence):
        command = [str(python), '-u', str(SAM3_WORKER), '--serve',
                   '--confidence', str(confidence)]
        if checkpoint:
            command += ['--checkpoint', str(checkpoint)]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True, bufsize=1)
        self.lines = Queue()
        self.errors = deque(maxlen=30)
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()
        try:
            if not self._response().get('ready'):
                raise RuntimeError('SAM 3 服务未完成初始化。')
        except Exception:
            self.close()
            raise

    def _read_stdout(self):
        for line in self.process.stdout:
            self.lines.put(line)
        self.lines.put(None)

    def _read_stderr(self):
        for line in self.process.stderr:
            self.errors.append(line)

    def _response(self, timeout=600.):
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError('SAM 3 响应超时。')
            try:
                line = self.lines.get(timeout=remaining)
            except Empty as exc:
                raise RuntimeError('SAM 3 响应超时。') from exc
            if line is None:
                raise RuntimeError(f'SAM 3 服务意外退出：{"".join(self.errors)[-1500:]}')
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'error' in payload:
                raise RuntimeError(f'SAM 3 推理失败：{payload["error"]}')
            return payload

    def generate(self, rgb, output, prompt):
        output = Path(output)
        with tempfile.TemporaryDirectory() as directory:
            rgb_path = Path(directory) / 'rgb.png'
            Image.fromarray(rgb).save(rgb_path)
            request = {'rgb': str(rgb_path), 'output': str(output), 'prompt': prompt}
            self.process.stdin.write(json.dumps(request) + '\n')
            self.process.stdin.flush()
            response = self._response()
        return [output / name for name in response['files']]

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        try:
            self.process.stdin.close()
        except OSError:
            pass
        self.process.stdout.close()
        self.process.stderr.close()


def select_mask(paths, index):
    if not paths:
        raise ValueError('SAM 3 未找到目标物体。')
    if index is None and len(paths) != 1:
        raise ValueError(f'SAM 3 找到 {len(paths)} 个目标；请指定 --instance-index。')
    chosen = index or 1
    if not 1 <= chosen <= len(paths):
        raise ValueError(f'目标序号 {chosen} 超出 1..{len(paths)}。')
    return np.asarray(Image.open(paths[chosen - 1])) > 0


def capture(url, camera_frame, max_age_s):
    rgb, depth, K, _, header = fetch_frame(url, with_metadata=True)
    if header.get('camera_info', {}).get('frame_id') != camera_frame:
        raise ValueError('RGB optical frame 与标定相机 frame 不一致。')
    age = header.get('rgb_age_s')
    if not isinstance(age, (int, float)) or not np.isfinite(age) or not 0 <= age <= max_age_s:
        raise ValueError('RGB 帧年龄未知或过旧；请更新并检查 HTTP 采集服务。')
    stamp = float(header['local_receive_stamp']) - age
    return rgb, depth, K, stamp, float(header['rgb_stamp'])


def capture_retry(args, timeout=2.):
    deadline = time.monotonic() + timeout
    while True:
        try:
            return capture(args.camera_url, args.camera_frame, args.max_frame_age)
        except (ValueError, OSError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(.05)


def verify_target(reference, current, mask, reference_depth, depth):
    if current.shape != reference.shape or mask.shape != depth.shape:
        raise ValueError('图像尺寸变化；需要重新生成目标掩码。')
    diff = np.abs(current[mask].astype(np.int16) - reference[mask].astype(np.int16))
    if diff.size == 0 or diff.mean() > 8 or (diff > 25).mean() > .05:
        raise ValueError('目标区域发生变化；需要重新生成掩码并确认目标。')
    valid = mask & (reference_depth > 0) & (depth > 0)
    if valid.sum() < 50 or np.median(np.abs(depth[valid].astype(np.int32) -
                                            reference_depth[valid].astype(np.int32))) > 20:
        raise ValueError('目标深度发生变化；需要重新生成掩码。')


def run(args, estimator=None, sam3_client=None, cache=None):
    reference, reference_depth, _, _, _ = capture_retry(args)
    mask = None
    if cache is not None and cache.get('mask') is not None:
        try:
            verify_target(cache['rgb'], reference, cache['mask'],
                          cache['depth'], reference_depth)
            mask = cache['mask']
        except ValueError:
            cache.clear()
    with tempfile.TemporaryDirectory() as directory:
        if mask is None:
            output = Path(directory) / 'masks'
            masks = (sam3_client.generate(reference, output, args.prompt)
                     if sam3_client is not None else
                     generate_masks(reference, output, args.sam3_python,
                                    args.prompt, args.sam3_checkpoint, args.confidence))
            mask = select_mask(masks, args.instance_index)
        if mask.shape != reference.shape[:2]:
            raise ValueError('SAM 3 掩码与 RGB 尺寸不一致。')
        if cache is not None:
            cache.update(rgb=reference, depth=reference_depth, mask=mask)
        if estimator is None:
            estimator = PoseEstimator(args.checkpoint, radio_repository=args.radio_repository,
                                      radio_checkpoint=args.radio_checkpoint, device=args.device)
        started = time.monotonic()
        published = 0
        last_rgb_stamp = None
        while time.monotonic() - started < args.duration and published < args.max_frames:
            try:
                rgb, depth, K, stamp, rgb_stamp = capture_retry(args)
                if rgb_stamp == last_rgb_stamp:
                    time.sleep(args.interval)
                    continue
                verify_target(reference, rgb, mask, reference_depth, depth)
                result = estimator.predict(rgb, depth, K, mask, .001, return_shape=False)
                age = time.time() - stamp
                if not -.05 <= age <= args.max_pose_age:
                    raise ValueError(f'推理完成时图像已过期：{age:.3f} s。')
                pose = {'stamp': stamp, 'source_rgb_stamp': rgb_stamp,
                        'frame_id': args.frame_id,
                        'position_m': np.asarray(result['translation_m']).tolist(),
                        'quaternion_xyzw': Rotation.from_matrix(result['rotation']).as_quat().tolist()}
                print(json.dumps({'pose': pose}, allow_nan=False), flush=True)
                last_rgb_stamp = rgb_stamp
                published += 1
            except ValueError as exc:
                # 场景变化后绝不沿用旧掩码；上游抓取程序将等待新观测并超时。
                if cache is not None:
                    cache.clear()
                print(json.dumps({'error': str(exc)}, ensure_ascii=False), flush=True)
                return 1
            time.sleep(args.interval)
    return 0


def serve(args):
    estimator = PoseEstimator(args.checkpoint, radio_repository=args.radio_repository,
                              radio_checkpoint=args.radio_checkpoint, device=args.device)
    client = Sam3Client(args.sam3_python, args.sam3_checkpoint, args.confidence)
    cache = {}
    print(json.dumps({'ready': True}), flush=True)
    try:
        for line in sys.stdin:
            try:
                if json.loads(line).get('request'):
                    run(args, estimator, client, cache)
            except (ValueError, RuntimeError, OSError, KeyError) as exc:
                cache.clear()
                print(json.dumps({'error': str(exc)}, ensure_ascii=False), flush=True)
            finally:
                print(json.dumps({'done': True}), flush=True)
    finally:
        client.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera-url', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--radio-repository', required=True)
    parser.add_argument('--radio-checkpoint', required=True)
    parser.add_argument('--sam3-python', required=True)
    parser.add_argument('--sam3-checkpoint', required=True)
    parser.add_argument('--prompt', default='water bottle')
    parser.add_argument('--confidence', type=float, default=.5)
    parser.add_argument('--instance-index', type=int)
    parser.add_argument('--camera-frame', default='camera_color_optical_frame')
    parser.add_argument('--frame-id', default='luna_head_calibrated_optical')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-frame-age', type=float, default=.15)
    parser.add_argument('--max-pose-age', type=float, default=.5)
    parser.add_argument('--max-frames', type=int, default=60)
    parser.add_argument('--duration', type=float, default=20.)
    parser.add_argument('--interval', type=float, default=.15)
    parser.add_argument('--serve', action='store_true', help='Keep F2G-Pose and SAM 3 loaded between requests')
    args = parser.parse_args(argv)
    if (not args.prompt.strip() or not 0 <= args.confidence <= 1 or
            args.instance_index is not None and args.instance_index < 1 or
            args.max_frames < 1 or not all(np.isfinite(v) and v > 0 for v in
                                         (args.max_frame_age, args.max_pose_age,
                                          args.duration, args.interval))):
        parser.error('目标、置信度、序号或采集时限无效。')
    try:
        if args.serve:
            return serve(args)
        return run(args)
    except (ValueError, RuntimeError, OSError) as exc:
        print(json.dumps({'error': str(exc)}, ensure_ascii=False), flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
