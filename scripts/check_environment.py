"""Check the local Python, CUDA, model assets and preview tools.

python scripts/check_environment.py [--assets] [--render] [--json report.json]
--assets also checks downloaded files, SAM 3 imports and the Blender VRM Add-on.
--render checks OpenGL rendering and FFmpeg.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run_check(name, function):
    try:
        return dict(name=name, ok=True, detail=str(function()))
    except Exception as exc:
        return dict(name=name, ok=False, detail=f'{type(exc).__name__}: {exc}')


def python_check():
    if sys.version_info[:2] != (3, 12) or platform.architecture()[0] != '64bit':
        raise RuntimeError('Use Python 3.12 x64 for the pinned release environment.')
    return f'{platform.system()} {platform.machine()}, Python {platform.python_version()}'


def device_check():
    """The device hairdeck.dev.select() picks (HAIRDECK_DEVICE), and a bf16 matmul on it."""
    from hairdeck import dev
    import torch
    import torchvision
    device = dev.select()
    if device.type == 'cpu' and not torch.cuda.is_available() and not torch.backends.mps.is_available():
        raise RuntimeError('No CUDA or Apple MPS device: install the CUDA PyTorch build, or set HAIRDECK_DEVICE=cpu (very slow).')
    x = torch.ones((16, 16), device=device, dtype=torch.bfloat16)
    y = x @ x
    dev.sync(device)
    if not bool(torch.isfinite(y).all()):
        raise RuntimeError(f'bfloat16 matmul on {device} returned non-finite values.')
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        name = props.name
        mem = f'{props.total_memory / 2**30:.1f} GiB'
    else:
        name, mem = ('Apple MPS', 'unified memory') if device.type == 'mps' else ('CPU', 'system memory')
    return f'torch {torch.__version__}, torchvision {torchvision.__version__}; {device.type}: {name}, {mem}.'


def dependency_check():
    names = ('gradio', 'pyrender', 'PyOpenGL', 'numpy', 'transformers', 'huggingface-hub')

    def version(name):  # gradio only for the demo, pyrender only for the previews
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return 'absent'
    versions = ', '.join(f'{name} {version(name)}' for name in names)
    try:
        import triton
        versions += f', triton {triton.__version__}'
    except ModuleNotFoundError:  # absent on macOS, where SAM 3 only needs the import (hairdeck/vendor/triton)
        versions += ', triton (absent)'
    return versions


def assets_check():
    from hairdeck import paths
    config = json.loads((ROOT / 'configs/released_models.json').read_text(encoding='utf-8'))
    names = [config[key]['checkpoint'] for key in ('card_vae', 'chain_dit', 'image_dit')]
    names += config['inference_data']['files'] + config['avatar_body']['files']
    names += ['ckpt/card_vae/stats.json', 'ckpt/card_vae/latent_stats_head.json']
    files = [paths.resolve(name) for name in names]
    files += [paths.weights(*parts) for parts in (
        ('sam3', 'sam3.pt'), ('dinov3', 'model.safetensors'),
        ('birefnet_matting', 'model.safetensors'), ('birefnet_matting', 'birefnet.py'),
        ('birefnet_matting', 'BiRefNet_config.py'))]
    files += [os.path.join(paths.dinov3_repo(), 'hubconf.py')]
    missing = [p for p in files if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError('Run scripts/setup/download_weights.py. Missing: ' + ', '.join(missing))
    return f'{len(files)} required files present.'


def sam3_check():
    from hairdeck.prep.sam3_ground import _sam3
    _sam3()
    return 'SAM 3 image builder and processor imported.'


def blender_check():
    from hairdeck.blender_runtime import blender_cmd, blender_env
    expr = ("import bpy; assert hasattr(bpy.types.Armature, 'vrm_addon_extension'), "
            "'VRM Add-on is not enabled'; print('HAIRDECK_BLENDER_READY')")
    result = subprocess.run(blender_cmd() + ['-b', '--python-exit-code', '1', '--python-expr', expr],
                            env=blender_env(), capture_output=True, text=True, errors='replace', timeout=60)
    if result.returncode or 'HAIRDECK_BLENDER_READY' not in result.stdout:
        raise RuntimeError('Blender/VRM Add-on check failed: ' + (result.stdout + result.stderr)[-1500:])
    return 'Blender started with the VRM Add-on enabled.'


def render_check():
    # A fresh process ensures backend selection occurs before any OpenGL import.
    code = '''
import os
import numpy as np
import trimesh
from hairdeck.vrm.render import _pyrender
p = _pyrender()
scene = p.Scene(ambient_light=[1, 1, 1])
scene.add(p.Mesh.from_trimesh(trimesh.creation.icosphere(subdivisions=0, radius=0.5)))
pose = np.eye(4); pose[2, 3] = 2
scene.add(p.PerspectiveCamera(yfov=0.8), pose=pose)
r = p.OffscreenRenderer(32, 32)
try:
    color, depth = r.render(scene)
    assert color.shape[:2] == (32, 32) and np.any(depth > 0), 'No mesh pixels rendered'
finally:
    r.delete()
print('32x32 mesh rendered using ' + os.environ.get('PYOPENGL_PLATFORM', 'Pyglet'))
'''
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True,
                            text=True, errors='replace', timeout=60)
    if result.returncode:
        raise RuntimeError((result.stdout + result.stderr)[-1500:])
    return result.stdout.strip()


def ffmpeg_check():
    from hairdeck.vis.turn_preview import ffmpeg_executable
    exe = ffmpeg_executable()
    result = subprocess.run([exe, '-version'], capture_output=True, text=True, errors='replace', timeout=15)
    if result.returncode:
        raise RuntimeError((result.stdout + result.stderr)[-1000:])
    return result.stdout.splitlines()[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--json', dest='report', help='save an environment report')
    args = parser.parse_args()
    checks = [('Python', python_check), ('Device / bfloat16', device_check), ('Dependencies', dependency_check)]
    if args.assets:
        checks += [('Downloaded assets', assets_check), ('SAM 3 import', sam3_check), ('Blender / VRM', blender_check)]
    if args.render:
        checks += [('Preview rendering', render_check), ('FFmpeg', ffmpeg_check)]
    results = []
    for name, function in checks:
        result = run_check(name, function)
        results.append(result)
        print(f"[{'PASS' if result['ok'] else 'FAIL'}] {name}: {result['detail']}", flush=True)
    report = dict(checks=results, ok=all(r['ok'] for r in results), scope='environment')
    if args.report:
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Environment checks passed.' if report['ok'] else 'Environment checks failed; see the results above.')
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
