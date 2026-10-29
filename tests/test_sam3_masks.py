import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from f2g_pose.demo import confirmed_mask, image_digest
from f2g_pose.sam3 import generate_masks


WORKER = Path(__file__).resolve().parents[1] / 'src' / 'f2g_pose' / 'sam3_worker.py'
spec = importlib.util.spec_from_file_location('sam3_masks_worker', WORKER)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def test_sam3_instances_are_separate_binary_masks_sorted_left_to_right(tmp_path):
    masks = np.zeros((3, 1, 12, 20), dtype=bool)
    masks[0, 0, 2:8, 14:18] = True
    masks[1, 0, 1:9, 2:6] = True
    records = worker.save_masks(tmp_path / 'masks', masks, [0.7, 0.9, 0.8], (12, 20), 'water bottle')
    assert [r['score'] for r in records] == [0.9, 0.7]
    left = np.asarray(Image.open(tmp_path / 'masks' / 'mask_01.png'))
    right = np.asarray(Image.open(tmp_path / 'masks' / 'mask_02.png'))
    assert left.shape == right.shape == (12, 20)
    assert set(np.unique(left)) == {0, 255}
    assert np.all(left[:, 14:18] == 0) and np.all(right[:, 2:6] == 0)
    with pytest.raises(FileExistsError):
        worker.save_masks(tmp_path / 'masks', masks, [0.7, 0.9, 0.8], (12, 20), 'water bottle')


def test_sam3_empty_result_and_bad_shape(tmp_path):
    records = worker.save_masks(tmp_path / 'none', np.empty((0, 1, 12, 20)), [], (12, 20), 'bottle')
    assert records == []
    with pytest.raises(ValueError, match='match RGB'):
        worker.save_masks(tmp_path / 'bad', np.ones((1, 10, 20)), [0.9], (12, 20), 'bottle')


def test_sam3_runner_isolated_from_main_python(monkeypatch, tmp_path):
    import f2g_pose.sam3 as sam3
    rgb = np.zeros((12, 20, 3), np.uint8)
    def fake_run(command, **kwargs):
        assert command[0] == '/sam3/python'
        assert command[1] == str(WORKER)
        assert np.asarray(Image.open(command[command.index('--rgb') + 1])).shape == rgb.shape
        worker.save_masks(command[command.index('--output') + 1], np.ones((1, 12, 20)),
                          [0.95], (12, 20), 'water bottle')
    monkeypatch.setattr(sam3.subprocess, 'run', fake_run)
    paths = generate_masks(rgb, tmp_path / 'auto', '/sam3/python')
    assert len(paths) == 1 and paths[0].is_file()
    with pytest.raises(FileExistsError):
        generate_masks(rgb, tmp_path / 'auto', '/sam3/python')


def test_confirmed_auto_mask_binds_to_current_image_even_with_uploaded_mask(tmp_path):
    rgb = np.zeros((12, 20, 3), np.uint8)
    uploaded = tmp_path / 'uploaded.png'
    Image.fromarray(np.zeros((12, 20), np.uint8)).save(uploaded)
    mask = np.ones((12, 20), bool)
    state = {'digest': image_digest(rgb), 'mask': mask, 'confirmed': True}
    assert confirmed_mask(rgb, str(uploaded), state).all()
    state['confirmed'] = False
    with pytest.raises(ValueError, match='confirm'):
        confirmed_mask(rgb, str(uploaded), state)
    changed = rgb.copy(); changed[0, 0] = 1
    assert not confirmed_mask(changed, str(uploaded), state).any()


def test_demo_auto_select_confirm_flow(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import f2g_pose.sam3 as sam3
    from f2g_pose.demo import build_demo

    rgb = np.zeros((12, 20, 3), np.uint8)
    def fake_generate(image, output, python, prompt, checkpoint):
        assert python == '/sam3/python' and prompt == 'water bottle'
        output.mkdir()
        path = output / 'mask_01.png'
        Image.fromarray(np.ones((12, 20), np.uint8) * 255).save(path)
        return [path]
    monkeypatch.setattr(sam3, 'generate_masks', fake_generate)
    app = build_demo(SimpleNamespace(device='cuda:0'), SimpleNamespace(
        sam_checkpoint=None, sam3_python='/sam3/python', sam3_checkpoint=None,
        output_root=tmp_path))
    functions = {f.fn.__name__: f.fn for f in app.fns.values()}
    available, options, files, first_state, first_preview, _ = functions['find_all'](rgb, 'water bottle')
    assert len(files) == 1 and options['choices'][0][1] == files[0]
    assert options['value'] == files[0] and first_preview.shape == rgb.shape
    assert not first_state['confirmed']
    preview, state, _ = functions['select_auto'](rgb, files[0], available)
    assert preview.shape == rgb.shape and not state['confirmed']
    state, _ = functions['confirm_target'](rgb, state)
    assert confirmed_mask(rgb, None, state).all()
    app.close()
