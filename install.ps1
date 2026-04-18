# instagram-ai-agent — Windows one-shot installer (PowerShell).
#
# Usage (PowerShell as regular user, from any directory):
#   iwr -useb https://raw.githubusercontent.com/alsk1992/instagram-ai-agent/main/install.ps1 | iex
#
# What it does:
#   1. Checks Python 3.11+ and ffmpeg are on PATH
#   2. Clones the repo into .\instagram-ai-agent (or cd's in if already cloned)
#   3. Creates .venv, pip-installs the agent + Playwright chromium + fonts
#   4. Prints next-step guidance
#
# Idempotent — rerunning refreshes deps.

$ErrorActionPreference = "Stop"

$REPO_URL = "https://github.com/alsk1992/instagram-ai-agent.git"
$REPO_DIR = "instagram-ai-agent"

function Say  { param([string]$m) Write-Host "==> $m" -ForegroundColor Cyan }
function Ok   { param([string]$m) Write-Host "✓ $m" -ForegroundColor Green }
function Warn { param([string]$m) Write-Host "⚠  $m" -ForegroundColor Yellow }
function Die  { param([string]$m) Write-Host "✗  $m" -ForegroundColor Red; exit 1 }

# ─── Prerequisites ───
Say "Checking prerequisites"

$pythonBin = $null
foreach ($candidate in @("python3.13", "python3.12", "python3.11", "python", "python3", "py")) {
    try {
        $ver = & $candidate -c "import sys; v=sys.version_info; exit(0 if (v.major, v.minor) >= (3, 11) else 1); print(f'{v.major}.{v.minor}')" 2>$null
        if ($LASTEXITCODE -eq 0) {
            $pythonBin = $candidate
            break
        }
    } catch { continue }
}
if (-not $pythonBin) {
    Die @"
Python 3.11+ required.
      Install from https://python.org/downloads/
      Or: winget install Python.Python.3.12
      Or: choco install python
"@
}
$pyVer = & $pythonBin --version
Ok "Python: $pythonBin ($pyVer)"

$ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
$ffprobe = Get-Command ffprobe -ErrorAction SilentlyContinue
if (-not $ffmpeg -or -not $ffprobe) {
    Die @"
ffmpeg + ffprobe required for reels.
      winget install Gyan.FFmpeg
      OR: choco install ffmpeg
      OR: scoop install ffmpeg
      OR: download from https://ffmpeg.org/download.html#build-windows and add to PATH
"@
}
Ok "ffmpeg: $(($ffmpeg | Get-Item).VersionInfo.FileVersionRaw)"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Die "git required. Install from https://git-scm.com/download/win (or: winget install Git.Git)"
}

# ─── Clone / cd ───
if ((Test-Path "pyproject.toml") -and (Select-String -Path "pyproject.toml" -Pattern 'name = "instagram-ai-agent"' -Quiet)) {
    Say "Running from existing checkout — skipping clone"
    $REPO_DIR = (Get-Location).Path
} elseif (Test-Path "$REPO_DIR\.git") {
    Say "Updating existing clone in $REPO_DIR"
    Push-Location $REPO_DIR
    try { git pull --ff-only } catch { Warn "git pull failed — continuing with existing code" }
} else {
    Say "Cloning $REPO_URL → $REPO_DIR"
    git clone --depth 1 $REPO_URL $REPO_DIR
    Push-Location $REPO_DIR
}
Ok "Working in: $(Get-Location)"

# ─── venv + install ───
Say "Creating virtualenv in .venv"
if (-not (Test-Path ".venv")) {
    & $pythonBin -m venv .venv
    Ok "Created .venv"
} else {
    Ok "Reusing existing .venv"
}

$venvPython = ".\.venv\Scripts\python.exe"

Say "Installing instagram-ai-agent + dependencies (this is the slow step)"
& $venvPython -m pip install --upgrade pip --quiet
& $venvPython -m pip install -e "." --quiet
Ok "Package installed"

# ─── Playwright chromium ───
Say "Installing Playwright chromium (~200 MB download, 2–5 min on first run)"
& $venvPython -m playwright install chromium
Ok "Playwright ready"

# ─── Fonts ───
Say "Fetching commercial-safe fonts (SIL-OFL)"
New-Item -ItemType Directory -Path "data\fonts" -Force | Out-Null

function DownloadFont {
    param([string]$url, [string]$dest)
    if (-not (Test-Path $dest)) {
        try { Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing }
        catch { Warn "font fetch failed ($dest) — DejaVu fallback will be used" }
    }
}
DownloadFont "https://github.com/google/fonts/raw/main/ofl/archivoblack/ArchivoBlack-Regular.ttf" "data\fonts\Archivo Black.ttf"
DownloadFont "https://github.com/google/fonts/raw/main/ofl/inter/Inter%5Bopsz%2Cwght%5D.ttf" "data\fonts\Inter.ttf"
Ok "Fonts ready"

# ─── Default meme backgrounds ───
if (Test-Path "scripts\gen_default_assets.py") {
    Say "Regenerating default meme backgrounds"
    try { & $venvPython "scripts\gen_default_assets.py" | Out-Null } catch {}
    Ok "Assets ready"
}

# ─── Next steps ───
Write-Host "`n✓ Install complete.`n" -ForegroundColor Green
Write-Host "Next steps:" -ForegroundColor White
Write-Host "  cd $REPO_DIR" -ForegroundColor Gray
Write-Host "  .\.venv\Scripts\Activate.ps1`n" -ForegroundColor Gray
Write-Host "  1. ig-agent init              # interactive wizard"
Write-Host "  2. ig-agent login             # verify IG credentials"
Write-Host "  3. ig-agent generate -n 3     # make 3 posts"
Write-Host "  4. ig-agent review            # approve them"
Write-Host "  5. ig-agent drain             # post NOW"
Write-Host "  6. ig-agent run               # start the full agent`n"
Write-Host "Need at least one free LLM key: OPENROUTER_API_KEY (https://openrouter.ai/keys)"
Write-Host "Docs: https://github.com/alsk1992/instagram-ai-agent"
