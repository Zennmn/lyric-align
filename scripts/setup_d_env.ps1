param(
    [string]$Root = (Resolve-Path (Join-Path $PSScriptRoot ".."))
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path $Root).Path
$env:UV_CACHE_DIR = Join-Path $Root ".cache\uv"
$env:UV_PYTHON_INSTALL_DIR = Join-Path $Root ".python"
$env:PIP_CACHE_DIR = Join-Path $Root ".cache\pip"
$env:HF_HOME = Join-Path $Root ".cache\huggingface"
$env:HF_HUB_CACHE = Join-Path $env:HF_HOME "hub"
$env:HF_DATASETS_CACHE = Join-Path $env:HF_HOME "datasets"
$env:MODELSCOPE_CACHE = Join-Path $Root ".cache\modelscope"
$env:TORCH_HOME = Join-Path $Root ".cache\torch"
$env:MELBAND_ROFORMER_MODELS_PATH = Join-Path $Root "models\melband"

New-Item -ItemType Directory -Force -Path @(
    $env:UV_CACHE_DIR,
    $env:UV_PYTHON_INSTALL_DIR,
    $env:PIP_CACHE_DIR,
    $env:HF_HUB_CACHE,
    $env:HF_DATASETS_CACHE,
    $env:MODELSCOPE_CACHE,
    $env:TORCH_HOME,
    $env:MELBAND_ROFORMER_MODELS_PATH,
    (Join-Path $Root "models\qwen"),
    (Join-Path $Root "models\hubertfa"),
    (Join-Path $Root "audio_cache"),
    (Join-Path $Root "outputs")
) | Out-Null

uv python install 3.12.13
if (-not (Test-Path (Join-Path $Root ".venv\Scripts\python.exe"))) {
    uv venv --python 3.12.13 (Join-Path $Root ".venv")
}

$Python = Join-Path $Root ".venv\Scripts\python.exe"
uv pip install --python $Python torch torchaudio --index-url "https://download.pytorch.org/whl/cu128"
uv pip install --python $Python -e "$Root[pipeline,hubertfa,dev]"

$ffmpeg = & $Python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"
$ffmpegDir = Split-Path -Parent $ffmpeg.Trim()
if ($env:PATH -notlike "*$ffmpegDir*") {
    $env:PATH = "$ffmpegDir$([IO.Path]::PathSeparator)$env:PATH"
}

Write-Output "Environment ready: $Python"
Write-Output "HF cache: $env:HF_HOME"
Write-Output "Mel-Band cache: $env:MELBAND_ROFORMER_MODELS_PATH"
Write-Output "ffmpeg: $ffmpeg"
