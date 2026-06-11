$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$WorkDir = Join-Path $Root "services\llm"
Set-Location $WorkDir

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "   NovaCare - Starting LLM Backend" -ForegroundColor White
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

$VenvPath = Join-Path $Root "services\llm\venv"
if (-not (Test-Path $VenvPath)) {
    Write-Host "[*] Detecting available Python versions..." -ForegroundColor Yellow
    $hasPy10 = $false
    try {
        $py10Val = & py -3.10 -c "print('ok')" -ErrorAction SilentlyContinue
        if ($py10Val -eq "ok") { $hasPy10 = $true }
    } catch {}
    
    $hasPy12 = $false
    try {
        $py12Val = & py -3.12 -c "print('ok')" -ErrorAction SilentlyContinue
        if ($py12Val -eq "ok") { $hasPy12 = $true }
    } catch {}

    if ($hasPy10) {
        Write-Host "[*] Creating virtual environment with Python 3.10..." -ForegroundColor Cyan
        & py -3.10 -m venv $VenvPath
    } elseif ($hasPy12) {
        Write-Host "[*] Creating virtual environment with Python 3.12..." -ForegroundColor Cyan
        & py -3.12 -m venv $VenvPath
    } else {
        Write-Host "[*] Creating virtual environment with default Python..." -ForegroundColor Cyan
        python -m venv $VenvPath
    }

    & "$VenvPath\Scripts\Activate.ps1"
    Write-Host "[*] Installing dependencies..." -ForegroundColor Yellow
    python -m pip install -r requirements.txt
    Write-Host "[OK] Dependencies installed" -ForegroundColor Green
} else {
    & "$VenvPath\Scripts\Activate.ps1"
    Write-Host "[OK] LLM Backend venv activated" -ForegroundColor Green
    Write-Host "[*] Ensuring dependencies are installed..." -ForegroundColor Yellow
    python -m pip install -r requirements.txt
}

$RootEnv = Join-Path $Root ".env"
if (-not (Test-Path '.env') -and (Test-Path $RootEnv)) {
    Copy-Item $RootEnv '.env'
    Write-Host "[OK] Copied unified .env from root to LLM backend directory" -ForegroundColor Green
} elseif (-not (Test-Path '.env')) {
    Write-Host "[!] WARNING: No .env file found!" -ForegroundColor Red
    Write-Host "[!] Create .env with OLLAMA_MODEL and/or HUGGINGFACE_API_KEY" -ForegroundColor Yellow
}

Write-Host "[*] Starting Flask on port 5000..." -ForegroundColor Cyan
python start_server.py
