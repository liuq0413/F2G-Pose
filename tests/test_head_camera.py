import json
import struct

import cv2
import numpy as np
import pytest

from f2g_pose.head_camera import decode_frame


def packet(*, aligned=True, delta=0.01, depth_width=12, camera_info=True):
    rgb = np.full((8, 12, 3), (20, 100, 200), np.uint8)
    depth = np.full((8, depth_width), 1200, np.uint16)
    ok_rgb, jpeg = cv2.imencode('.jpg', rgb)
    ok_depth, png = cv2.imencode('.png', depth)
    assert ok_rgb and ok_depth
    metadata = dict(rgb_stamp=1., depth_stamp=1. + delta, depth_unit='mm',
                    depth_aligned_to_color=aligned,
                    camera_info={'K': [100, 0, 6, 0, 100, 4, 0, 0, 1],
                                 'width': 12, 'height': 8} if camera_info else None,
                    rgb_bytes=len(jpeg), depth_bytes=len(png))
    header = json.dumps(metadata).encode()
    return struct.pack('!I', len(header)) + header + jpeg.tobytes() + png.tobytes()


def test_aligned_frame_provides_color_depth_and_intrinsics():
    rgb, depth, K, delta = decode_frame(packet())
    assert rgb.shape == (8, 12, 3)
    assert rgb[0, 0, 0] > rgb[0, 0, 2]
    assert depth.dtype == np.uint16 and np.all(depth == 1200)
    assert K[0, 0] == 100 and K[1, 2] == 4
    assert delta == pytest.approx(.01)


@pytest.mark.parametrize('kwargs', [dict(aligned=False), dict(delta=.5),
                                  dict(depth_width=10), dict(camera_info=False)])
def test_unusable_frame_is_rejected(kwargs):
    with pytest.raises(ValueError):
        decode_frame(packet(**kwargs))
