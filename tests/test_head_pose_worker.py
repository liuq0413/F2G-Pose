from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from f2g_pose import head_pose_worker as worker


def test_target_selection_requires_explicit_index_for_multiple_bottles(tmp_path):
    paths = []
    for index in range(2):
        path = tmp_path / f'{index}.png'
        Image.fromarray(np.ones((8, 12), np.uint8) * 255).save(path)
        paths.append(path)
    with pytest.raises(ValueError, match='--instance-index'):
        worker.select_mask(paths, None)
    with pytest.raises(ValueError, match='超出'):
        worker.select_mask(paths, 3)
    assert worker.select_mask(paths, 2).all()


def test_mask_reuse_rejects_target_motion():
    ref = np.full((8, 12, 3), 100, np.uint8)
    depth = np.full((8, 12), 1000, np.uint16)
    mask = np.zeros((8, 12), bool)
    mask[2:7, 1:11] = True
    worker.verify_target(ref, ref.copy(), mask, depth, depth.copy())
    changed = ref.copy()
    changed[mask] = 200
    with pytest.raises(ValueError, match='目标区域'):
        worker.verify_target(ref, changed, mask, depth, depth)
    with pytest.raises(ValueError, match='目标深度'):
        worker.verify_target(ref, ref.copy(), mask, depth, depth + 100)


def test_capture_uses_local_arrival_time_and_rejects_old_frames(monkeypatch):
    header = {'camera_info': {'frame_id': 'camera_color_optical_frame'},
              'rgb_age_s': .04, 'local_receive_stamp': 100., 'rgb_stamp': 200.}
    monkeypatch.setattr(worker, 'fetch_frame', lambda *a, **k: (
        np.zeros((8, 12, 3), np.uint8), np.zeros((8, 12), np.uint16),
        np.eye(3), 0., header))
    _, _, _, stamp, source = worker.capture('http://camera/frame', 'camera_color_optical_frame', .15)
    assert stamp == pytest.approx(99.96) and source == 200.
    header['rgb_age_s'] = .3
    with pytest.raises(ValueError, match='过旧'):
        worker.capture('http://camera/frame', 'camera_color_optical_frame', .15)


def test_run_publishes_fresh_camera_pose_without_motion_commands(monkeypatch, tmp_path, capsys):
    rgb = np.full((8, 12, 3), 100, np.uint8)
    depth = np.full((8, 12), 1000, np.uint16)
    samples = iter([(rgb, depth, np.eye(3), 1., 10.),
                    (rgb, depth, np.eye(3), 2., 11.)])
    monkeypatch.setattr(worker, 'capture', lambda *a: next(samples))
    mask_path = tmp_path / 'mask.png'
    Image.fromarray(np.ones((8, 12), np.uint8) * 255).save(mask_path)
    monkeypatch.setattr(worker, 'generate_masks', lambda *a: [mask_path])
    monkeypatch.setattr(worker.time, 'time', lambda: 2.1)
    estimate = SimpleNamespace(predict=lambda *a, **k: {
        'translation_m': np.array([.2, .1, .8]), 'rotation': np.eye(3)})
    args = SimpleNamespace(camera_url='http://camera/frame', camera_frame='camera_color_optical_frame',
                           max_frame_age=.15, sam3_python='python', prompt='bottle',
                           sam3_checkpoint='checkpoint', confidence=.5, instance_index=None,
                           checkpoint='f2g', radio_repository='radio', radio_checkpoint='radio.ckpt',
                           device='cuda:0', duration=10., max_frames=1, max_pose_age=.5,
                           interval=.001, frame_id='luna_head_calibrated_optical')
    assert worker.run(args, estimate) == 0
    import json
    record = json.loads(capsys.readouterr().out)
    assert record['pose']['stamp'] == 2.
    assert record['pose']['position_m'] == [.2, .1, .8]
    assert record['pose']['quaternion_xyzw'] == [0., 0., 0., 1.]
