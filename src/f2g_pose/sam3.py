"""Call the optional SAM 3 environment without changing F2G-Pose dependencies."""
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from PIL import Image


WORKER = Path(__file__).with_name('sam3_worker.py')


def generate_masks(rgb, output, python, prompt='water bottle', checkpoint=None, confidence=0.5):
    if not python:
        raise ValueError('Set --sam3-python to the Python executable of the SAM 3 environment.')
    if not WORKER.is_file():
        raise FileNotFoundError(f'SAM 3 worker is missing: {WORKER}')
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        input_path = Path(tmp) / 'rgb.png'
        Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert('RGB').save(input_path)
        command = [str(python), str(WORKER), '--rgb', str(input_path), '--output', str(output),
                   '--prompt', prompt, '--confidence', str(confidence)]
        if checkpoint:
            command += ['--checkpoint', str(checkpoint)]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError('SAM 3 exceeded the 10-minute limit. Check its environment and checkpoint.') from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f'SAM 3 failed: {(exc.stderr or exc.stdout)[-1500:]}') from exc
    data = json.loads((output / 'masks.json').read_text())
    paths = [output / item['file'] for item in data['instances']]
    if any(not path.is_file() or path.parent != output for path in paths):
        raise RuntimeError('SAM 3 did not produce the expected instance masks.')
    return paths
