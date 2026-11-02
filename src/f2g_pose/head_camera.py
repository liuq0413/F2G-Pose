"""Read one aligned RGB-D frame from the head camera HTTP bridge."""
import json
import struct
import time
from urllib.request import ProxyHandler, build_opener

import cv2
import numpy as np


MAX_PACKET_BYTES = 16 * 1024 * 1024


def decode_frame(packet, max_delta_s=0.1, with_metadata=False):
    if not 4 <= len(packet) <= MAX_PACKET_BYTES:
        raise ValueError('Invalid head camera packet size.')
    header_size = struct.unpack('!I', packet[:4])[0]
    if not 0 < header_size <= 4096 or len(packet) < 4 + header_size:
        raise ValueError('Invalid head camera header size.')
    header = json.loads(packet[4:4 + header_size])
    rgb_size, depth_size = header['rgb_bytes'], header['depth_bytes']
    if (not all(isinstance(n, int) and n > 0 for n in (rgb_size, depth_size))
            or len(packet) != 4 + header_size + rgb_size + depth_size):
        raise ValueError('Head camera packet lengths do not match.')
    if header.get('depth_aligned_to_color') is not True:
        raise ValueError('Depth is not aligned to RGB. Start the bridge with the aligned_depth_to_color topic.')
    if header.get('depth_unit') != 'mm':
        raise ValueError('Head camera depth must be in millimeters.')
    delta = abs(float(header['rgb_stamp']) - float(header['depth_stamp']))
    if not np.isfinite(delta) or delta > max_delta_s:
        raise ValueError(f'RGB/depth timestamps differ by {delta:.3f} s; capture a synchronized pair.')
    start = 4 + header_size
    rgb_bgr = cv2.imdecode(np.frombuffer(packet[start:start + rgb_size], np.uint8), cv2.IMREAD_COLOR)
    depth = cv2.imdecode(np.frombuffer(packet[start + rgb_size:], np.uint8), cv2.IMREAD_UNCHANGED)
    if rgb_bgr is None or depth is None or depth.dtype != np.uint16 or depth.ndim != 2:
        raise ValueError('Head camera JPEG or 16-bit depth PNG is invalid.')
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    info = header.get('camera_info')
    if not isinstance(info, dict) or (info.get('width'), info.get('height')) != (rgb.shape[1], rgb.shape[0]):
        raise ValueError('Missing RGB camera intrinsics or camera_info dimensions do not match RGB.')
    from .preprocess import intrinsics_matrix
    K = intrinsics_matrix(np.asarray(info['K'], dtype=np.float32).reshape(3, 3))
    if depth.shape != rgb.shape[:2]:
        raise ValueError(f'RGB {rgb.shape[:2]} and depth {depth.shape} differ; align depth to color first.')
    return (rgb, depth, K, delta, header) if with_metadata else (rgb, depth, K, delta)


def fetch_frame(url, timeout=3, with_metadata=False):
    # 机器人局域网连接不应经过用户终端的 HTTP/SOCKS 代理。
    opener = build_opener(ProxyHandler({}))
    with opener.open(url, timeout=timeout) as response:
        packet = response.read(MAX_PACKET_BYTES + 1)
    received_at = time.time()
    result = decode_frame(packet, with_metadata=with_metadata)
    if with_metadata:
        result[-1]['local_receive_stamp'] = received_at
    return result
