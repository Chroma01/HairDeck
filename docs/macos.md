# macOS (Apple silicon) setup

HairDeck's release targets one NVIDIA CUDA GPU. On macOS the same code runs on Apple MPS through `hairdeck/dev.py`,
which every model call uses for device selection, bf16 autocast, synchronisation and seeded noise.

```bash
git clone https://github.com/hyz317/HairDeck.git && cd HairDeck
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt -r requirements-demo.txt
.venv/bin/pip install --no-deps pyrender==0.1.45    # previews only; its old PyOpenGL pin is not used
```

`torch==2.8.0` resolves to the macOS arm64 wheel with MPS: no CUDA and no Triton needed. SAM 3 imports Triton anyway, so
on macOS `hairdeck/vendor/triton` is put on `sys.path` when Triton is absent; its kernels only ever run on CUDA.

## Device setting

`HAIRDECK_DEVICE` selects the device:

| value | meaning |
|---|---|
| `favour` (default on macOS) | CUDA if present, else Apple MPS, else CPU |
| `cuda` (default elsewhere) | NVIDIA CUDA; an error if unavailable, as before |
| `mps` | Apple MPS, error if unavailable |
| `cpu` | CPU: correct, far too slow for the diffusion models |

## Weights

SAM 3 and DINOv3 are gated: accept the licences on [facebook/sam3](https://huggingface.co/facebook/sam3) and
[facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m), then

```bash
.venv/bin/hf auth login
export HAIRDECK_ROOT=~/HairDeck          # or write configs/paths.json, see configs/paths.example.json
.venv/bin/python scripts/setup/download_weights.py
```

The downloader has no managed Blender build for macOS (`blender` is not in its default set there): the GLB output does
not need Blender. VRM export and the head-turn preview do: install Blender 4.2 LTS from blender.org, enable the
[VRM Add-on](https://github.com/saturday06/VRM-Addon-for-Blender) in it, and point `HAIRDECK_BLENDER` at the executable
(or at its macOS app bundle binary, `/Applications/Blender.app/Contents/MacOS/Blender`).

The Add-on declares `blender_version_min = 4.2.0` / `blender_version_max = 5.3.0`, so newer Blenders work: this port was
verified on Blender 5.2.3 with Add-on 4.10.2 (VRM export, rest renders and the head-turn preview all pass):

```bash
curl -LO https://github.com/saturday06/VRM-Addon-for-Blender/releases/download/v4.10.2/VRM_Addon_for_Blender-Extension-4_10_2.zip
/Applications/Blender.app/Contents/MacOS/Blender -b --command extension install-file \
    -r user_default -e VRM_Addon_for_Blender-Extension-4_10_2.zip
export HAIRDECK_BLENDER=/Applications/Blender.app/Contents/MacOS/Blender
```

On macOS the downloader finishes by running `scripts/setup/patch_sam3_mps.py`. SAM 3 is written for CUDA in seven places
(`_setup_device_and_mode` only honours `"cuda"`; two position-encoding caches are built with `device="cuda"`; the fused
`aten::_addmm_activation` MLP and `_assert_async` checks have no MPS kernel; `pin_memory()` then `.to("mps")` is
rejected; an empty prompt reaches `grid_sample`). The script makes those device-agnostic, is idempotent and is limited
to those spots; re-run it (`--check` to only verify) after re-downloading `sam3_src`.

## Run

```bash
export HAIRDECK_ROOT=~/HairDeck
.venv/bin/python scripts/check_environment.py --assets --render    # diagnostics
.venv/bin/python scripts/image_to_hair.py character.png --out results/my_character --n 4
.venv/bin/python demo/app.py --port 7860                           # Gradio UI
```

Expect the diffusion sampling to be several times slower than on a CUDA card; 96 GB of unified memory means model
offload (`HAIRDECK_OFFLOAD`) never engages, so every model stays resident.

## Not ported

`scripts/test.py` (CHARM-protocol scan renders through EGL), `scripts/data.py` and `scripts/train/*` are unchanged: they
train on CUDA and still assume it.
