$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$WorkDir = Join-Path $Root "services\auth"
Set-Location $WorkDir

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "   NovaCare - Starting Auth Backend" -ForegroundColor White
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path 'venv')) {
    Write-Host "[*] Creating virtual environment (venv)..." -ForegroundColor Yellow
    python -m venv venv
    .\venv\Scripts\Activate.ps1
    Write-Host "[*] Installing dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt
    Write-Host "[OK] Dependencies installed" -ForegroundColor Green
} else {
    .\venv\Scripts\Activate.ps1
    Write-Host "[OK] Auth Backend venv activated" -ForegroundColor Green
    Write-Host "[*] Ensuring dependencies are installed..." -ForegroundColor Yellow
    pip install -r requirements.txt
}

if (-not (Test-Path '.env')) {
    if (Test-Path '.env.example') {
        Copy-Item '.env.example' '.env'
        Write-Host "[OK] Created default .env from .env.example" -ForegroundColor Green
    }
}

Write-Host "[*] Starting Auth Backend on port 8000..." -ForegroundColor Cyan
python run.py
