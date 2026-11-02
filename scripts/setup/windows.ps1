param([string]$Workspace = "", [string]$Python = "py")

$ErrorActionPreference = "Stop"
$RepoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "../.."))
if ($env:OS -ne "Windows_NT" -or $env:PROCESSOR_ARCHITECTURE -ne "AMD64") {
    throw "This installer requires Windows x64."
}
$env:PYTHONUTF8 = "1"
$Venv = Join-Path $RepoRoot ".venv"
$VenvPython = Join-Path $Venv "Scripts/python.exe"
if (-not (Test-Path $VenvPython)) {
    if (Test-Path $Venv) { throw "An incompatible .venv already exists. Move it aside before installing." }
    $LauncherArgs = @()
    if ($Python -eq "py") { $LauncherArgs = @("-3.12") }
    & $Python @LauncherArgs -m venv $Venv
    if ($LASTEXITCODE -ne 0) { throw "Could not create the environment. Install Python 3.12 x64 first." }
}

function Invoke-VenvPython {
    param([string[]]$PythonArguments)
    & $VenvPython @PythonArguments
    if ($LASTEXITCODE -ne 0) { throw "Python command failed (exit $LASTEXITCODE). See the output above." }
}

Invoke-VenvPython -PythonArguments @("-c", "import sys,struct; assert sys.version_info[:2] == (3,12) and struct.calcsize('P') == 8, 'Python 3.12 x64 is required'")
Invoke-VenvPython -PythonArguments @("-m", "pip", "install", "--upgrade", "pip")
Invoke-VenvPython -PythonArguments @("-m", "pip", "install", "torch==2.8.0", "torchvision==0.23.0", "--index-url", "https://download.pytorch.org/whl/cu128")
Invoke-VenvPython -PythonArguments @("-m", "pip", "install", "-r", (Join-Path $RepoRoot "requirements-windows.txt"), "-r", (Join-Path $RepoRoot "requirements-demo.txt"))
Invoke-VenvPython -PythonArguments @("-m", "pip", "install", "--no-deps", "pyrender==0.1.45")

$ConfigPath = Join-Path $RepoRoot "configs/paths.json"
if ($env:HAIRDECK_CONFIG) { $ConfigPath = [System.IO.Path]::GetFullPath($env:HAIRDECK_CONFIG) }
$Config = [PSCustomObject]@{}
if (Test-Path $ConfigPath) { $Config = Get-Content -Raw -Encoding UTF8 $ConfigPath | ConvertFrom-Json }
if ($Workspace -or -not $Config.root) {
    if (-not $Workspace) {
        $Workspace = $env:HAIRDECK_ROOT
        if (-not $Workspace) { $Workspace = Join-Path $RepoRoot "workspace" }
    }
    $Workspace = [System.IO.Path]::GetFullPath($Workspace)
    $Config | Add-Member -NotePropertyName root -NotePropertyValue $Workspace -Force
    $Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.Directory]::CreateDirectory([System.IO.Path]::GetDirectoryName($ConfigPath)) | Out-Null
    [System.IO.File]::WriteAllText($ConfigPath, ($Config | ConvertTo-Json -Depth 20), $Utf8NoBom)
    $env:HAIRDECK_ROOT = $Workspace
}

Invoke-VenvPython -PythonArguments @((Join-Path $RepoRoot "scripts/check_environment.py"))
Write-Host "Environment ready. Accept the SAM 3 / DINOv3 licenses, then run from the repository root:"
Write-Host '$env:PYTHONUTF8 = "1"'
Write-Host '.\.venv\Scripts\hf.exe auth login'
Write-Host '.\.venv\Scripts\python.exe scripts\setup\download_weights.py'
Write-Host '.\.venv\Scripts\python.exe scripts\check_environment.py --assets --render'
Write-Host '.\run_windows.bat'
