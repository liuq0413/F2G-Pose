import json
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


def test_prompt_list_accepts_repeated_and_comma_separated_values():
    args = SimpleNamespace(prompt=['water bottle,cup', 'cup;book'])
    assert worker.prompt_list(args) == ['water bottle', 'cup', 'book']


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


def test_run_publishes_name_and_pose_for_multiple_prompts(monkeypatch, tmp_path, capsys):
    rgb = np.full((8, 12, 3), 100, np.uint8)
    depth = np.full((8, 12), 1000, np.uint16)
    samples = iter([(rgb, depth, np.eye(3), 1., 10.),
                    (rgb, depth, np.eye(3), 2., 11.)])
    monkeypatch.setattr(worker, 'capture_retry', lambda *_: next(samples))
    monkeypatch.setattr(worker.time, 'time', lambda: 2.1)

    def fake_generate(_image, output, _python, prompt, _checkpoint, _confidence):
        output.mkdir()
        mask_path = output / 'mask_01.png'
        Image.fromarray(np.ones((8, 12), np.uint8) * 255).save(mask_path)
        return [mask_path]

    monkeypatch.setattr(worker, 'generate_masks', fake_generate)
    estimate = SimpleNamespace(predict=lambda *a, **k: {
        'translation_m': np.array([.2, .1, .8]), 'rotation': np.eye(3)})
    args = SimpleNamespace(camera_url='http://camera/frame', camera_frame='camera_color_optical_frame',
                           max_frame_age=.15, prompt=['bottle', 'cup'],
                           sam3_python='python', sam3_checkpoint='checkpoint', confidence=.5,
                           instance_index=None, checkpoint='f2g', radio_repository='radio',
                           radio_checkpoint='radio.ckpt', device='cuda:0', duration=10.,
                           max_frames=1, max_pose_age=.5, interval=.001,
                           frame_id='luna_head_calibrated_optical')
    assert worker.run(args, estimate) == 0
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [record['object_name'] for record in records] == ['bottle', 'cup']
    assert all(record['pose']['position_m'] == [.2, .1, .8] for record in records)


def test_missing_one_prompt_does_not_drop_other_object_pose(monkeypatch, tmp_path, capsys):
    rgb = np.full((8, 12, 3), 100, np.uint8)
    depth = np.full((8, 12), 1000, np.uint16)
    samples = iter([(rgb, depth, np.eye(3), 1., 10.),
                    (rgb, depth, np.eye(3), 2., 11.)])
    monkeypatch.setattr(worker, 'capture_retry', lambda *_: next(samples))
    monkeypatch.setattr(worker.time, 'time', lambda: 2.1)

    def fake_generate(_image, output, _python, prompt, _checkpoint, _confidence):
        if prompt == 'cup':
            raise ValueError('SAM 3 未找到目标物体。')
        output.mkdir()
        mask_path = output / 'mask_01.png'
        Image.fromarray(np.ones((8, 12), np.uint8) * 255).save(mask_path)
        return [mask_path]

    monkeypatch.setattr(worker, 'generate_masks', fake_generate)
    estimate = SimpleNamespace(predict=lambda *a, **k: {
        'translation_m': np.array([.2, .1, .8]), 'rotation': np.eye(3)})
    args = SimpleNamespace(camera_url='http://camera/frame', camera_frame='camera_color_optical_frame',
                           max_frame_age=.15, prompt=['water bottle', 'cup'],
                           sam3_python='python', sam3_checkpoint='checkpoint', confidence=.5,
                           instance_index=None, checkpoint='f2g', radio_repository='radio',
                           radio_checkpoint='radio.ckpt', device='cuda:0', duration=10.,
                           max_frames=1, max_pose_age=.5, interval=.001,
                           frame_id='luna_head_calibrated_optical')
    assert worker.run(args, estimate) == 0
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert records[0] == {'object_name': 'cup', 'error': 'SAM 3 未找到目标物体。'}
    assert records[1]['object_name'] == 'water bottle'
    assert records[1]['pose']['position_m'] == [.2, .1, .8]


def test_repeated_requests_reuse_mask_and_resegment_after_motion(monkeypatch, tmp_path, capsys):
    rgb = np.full((8, 12, 3), 100, np.uint8)
    moved = np.full_like(rgb, 200)
    depth = np.full((8, 12), 1000, np.uint16)
    samples = iter([
        (rgb, depth, np.eye(3), 1., 10.), (rgb, depth, np.eye(3), 2., 11.),
        (rgb, depth, np.eye(3), 3., 12.), (rgb, depth, np.eye(3), 4., 13.),
        (moved, depth, np.eye(3), 5., 14.), (moved, depth, np.eye(3), 6., 15.),
    ])
    monkeypatch.setattr(worker, 'capture_retry', lambda *_: next(samples))
    monkeypatch.setattr(worker.time, 'time', lambda: 6.1)
    calls = []

    class FakeSam3:
        def generate(self, image, output, prompt):
            calls.append(image.copy())
            output.mkdir()
            mask = output / 'mask_01.png'
            Image.fromarray(np.ones((8, 12), np.uint8) * 255).save(mask)
            return [mask]

    estimate = SimpleNamespace(predict=lambda *a, **k: {
        'translation_m': np.array([.2, .1, .8]), 'rotation': np.eye(3)})
    args = SimpleNamespace(camera_url='http://camera/frame', camera_frame='camera_color_optical_frame',
                           max_frame_age=.15, sam3_python='python', prompt='bottle',
                           sam3_checkpoint='checkpoint', confidence=.5, instance_index=None,
                           checkpoint='f2g', radio_repository='radio', radio_checkpoint='radio.ckpt',
                           device='cuda:0', duration=10., max_frames=1, max_pose_age=10.,
                           interval=.001, frame_id='luna_head_calibrated_optical')
    cache = {}
    for _ in range(3):
        assert worker.run(args, estimate, FakeSam3(), cache) == 0
    assert len(calls) == 2
    assert np.array_equal(calls[0], rgb) and np.array_equal(calls[1], moved)
    assert len(capsys.readouterr().out.splitlines()) == 3


def test_service_keeps_models_loaded_for_multiple_requests(monkeypatch, capsys):
    import io
    monkeypatch.setattr(worker.sys, 'stdin', io.StringIO('{"request":true}\n{"request":true}\n'))
    created = []
    requests = []

    def make_model(*args, **kwargs):
        model = object()
        created.append(model)
        return model

    class FakeSam3:
        def __init__(self, *args):
            created.append(self)

        def close(self):
            pass

    monkeypatch.setattr(worker, 'PoseEstimator', make_model)
    monkeypatch.setattr(worker, 'Sam3Client', FakeSam3)
    monkeypatch.setattr(worker, 'run', lambda args, model, client, cache:
                        requests.append((model, client, cache)))
    args = SimpleNamespace(checkpoint='f2g', radio_repository='radio',
                           radio_checkpoint='radio.ckpt', device='cuda:0',
                           sam3_python='python', sam3_checkpoint='sam3.pt', confidence=.5)
    worker.serve(args)
    messages = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert messages == [{'ready': True}, {'done': True}, {'done': True}]
    assert len(created) == 2 and len(requests) == 2
    assert requests[0][0] is requests[1][0] and requests[0][1] is requests[1][1]
    assert requests[0][2] is requests[1][2]
