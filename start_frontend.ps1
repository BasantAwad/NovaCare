$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$WorkDir = Join-Path $Root "apps\frontend"
Set-Location $WorkDir

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "   NovaCare - Starting Frontend" -ForegroundColor White
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path 'node_modules')) {
    Write-Host "[*] Installing npm dependencies..." -ForegroundColor Yellow
    npm install
    Write-Host "[OK] Dependencies installed" -ForegroundColor Green
} else {
    Write-Host "[OK] node_modules found" -ForegroundColor Green
}

if (-not (Test-Path '.env.local')) {
    Write-Host "[!] WARNING: No .env.local file found!" -ForegroundColor Red
    Write-Host "    Create .env.local referencing http://localhost:5000" -ForegroundColor Yellow
}

Write-Host "[*] Starting Next.js on port 3000..." -ForegroundColor Cyan
npm run dev -- --hostname 0.0.0.0
