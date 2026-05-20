$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$WorkDir = Join-Path $Root "services\llm-backend"
Set-Location $WorkDir

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "   NovaCare - Starting LLM Backend" -ForegroundColor White
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
    Write-Host "[OK] LLM Backend venv activated" -ForegroundColor Green
}

if (-not (Test-Path '.env')) {
    Write-Host "[!] WARNING: No .env file found!" -ForegroundColor Red
    Write-Host "[!] Create .env with OLLAMA_MODEL and/or HUGGINGFACE_API_KEY" -ForegroundColor Yellow
}

Write-Host "[*] Starting Flask on port 5000..." -ForegroundColor Cyan
python start_server.py
