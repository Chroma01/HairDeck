# Windows setup

Run HairDeck natively on Windows to generate hair cards and spring bones, use the Gradio demo, and export GLB meshes, VRM avatars and animation previews.

## Requirements

- Windows 10/11 x64 and Python 3.12 x64.
- An NVIDIA Ampere-or-newer GPU (compute capability 8.0+) and a driver compatible with CUDA 12.8.
- Microsoft Visual C++ 2015–2022 Redistributable (x64), used by Triton. Installation links are in the [Triton Windows instructions](https://github.com/triton-lang/triton-windows#6-vcredist).
- A desktop session with OpenGL for image and video previews.

The Windows pipeline has been run on an RTX 3070 Ti 8 GB. Automatic model offload manages GPU residency by moving inactive models to system RAM.

## Install

Open PowerShell and run:

```powershell
git clone https://github.com/hyz317/HairDeck.git
cd HairDeck
powershell -ExecutionPolicy Bypass -File .\scripts\setup\windows.ps1 -Workspace D:\HairDeckData
```

The script creates `.venv`, installs PyTorch 2.8 with CUDA 12.8 and the Windows/demo dependencies, and saves the workspace path in `configs/paths.json`. It includes the CUDA toolchain required by Triton and a bundled FFmpeg executable. The execution-policy option applies to this PowerShell process.

`-Workspace` sets the model/data directory. An existing configured root is used when the option is omitted; a fresh setup defaults to `workspace` inside the repository. `HAIRDECK_ROOT`, `HAIRDECK_CONFIG` and other path environment variables take precedence over the default config.

To select a Python installation explicitly, pass `-Python "C:\path\to\python.exe"`.

## Download models

For Hugging Face downloads, accept the licenses on [SAM 3](https://huggingface.co/facebook/sam3) and [DINOv3](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m), then log in and download:

```powershell
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\hf.exe auth login
.\.venv\Scripts\python.exe scripts\setup\download_weights.py
```

To download SAM 3 and DINOv3 from ModelScope instead, run:

```powershell
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\python.exe scripts\setup\download_weights.py --gated-source modelscope
```

Both sources use the same release weight checksums. The downloader also installs the Windows build of Blender 4.2 LTS and the VRM Add-on, with configuration and caches kept in the workspace.

Check CUDA, dependencies, model files, Blender, OpenGL and FFmpeg:

```powershell
.\.venv\Scripts\python.exe scripts\check_environment.py --assets --render --json windows-check.json
```

The JSON report contains the result of each environment check and can be attached to an issue when troubleshooting.

## Start Gradio

Double-click `run_windows.bat`, or run:

```powershell
.\run_windows.bat --port 7860
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860) in the Windows computer's browser. Upload an image, generate a hairstyle, inspect the GLB, and use **Export VRM** or **Preview motion**. The launcher uses `.venv` and enables Python UTF-8 mode for paths and logs.

## Command-line generation and export

```powershell
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\python.exe scripts\image_to_hair.py character.png --out results\my_character --n 1
.\.venv\Scripts\python.exe -m hairdeck.vrm.export results\my_character\character 0
```

Replace `character.png` with your image. Results are saved in a subdirectory named after that image. For a VRM-only export:

```powershell
.\.venv\Scripts\python.exe -m hairdeck.vrm.export results\my_character\character 0 vrm --no-render
```

## Model offload

Windows uses `HAIRDECK_OFFLOAD=auto` to select model residency from the GPU's total and available memory. SAM 3, BiRefNet and hair generation run in stages, with inactive model weights stored in system RAM. The selected models move to CUDA for their stage and return to CPU after the request.

Set the mode before launching:

```powershell
$env:HAIRDECK_OFFLOAD = 'auto' # automatic GPU memory management
# 'on': use stage offload for every request; 'off': keep models on the GPU
.\run_windows.bat --port 7860
```

Python callers can use `Pipeline(offload='auto')` or select a GPU with `Pipeline(device='cuda:1', offload='auto')`. Generation and built-in preprocessing use the selected GPU. Linux defaults to `off` and accepts the same overrides. Residency transitions appear in the server log and in the `memory` field of each result's `info.json`.

## Blender and preview configuration

To use your own Blender 4.2 LTS installation with the VRM Add-on enabled:

```powershell
$env:HAIRDECK_BLENDER = 'C:\Program Files\Blender Foundation\Blender 4.2\blender.exe'
```

HairDeck then uses that installation and its add-on settings. `HAIRDECK_BLENDER` also accepts a JSON array for commands with extra arguments.

Windows previews use Pyglet/OpenGL. Set `HAIRDECK_GL` to `pyglet`, `egl` or `osmesa` to select a backend for your environment. Linux defaults to EGL.

FFmpeg is selected from `HAIRDECK_FFMPEG`, then `PATH`, then the bundled `imageio-ffmpeg` executable. To choose a build:

```powershell
$env:HAIRDECK_FFMPEG = 'C:\Tools\ffmpeg\bin\ffmpeg.exe'
```
