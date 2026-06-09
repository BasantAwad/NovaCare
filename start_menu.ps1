# Clear Host for premium display
Clear-Host

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

function Write-Banner {
    Write-Host ""
    Write-Host "  ============================================" -ForegroundColor Cyan
    Write-Host "         NovaCare Interactive Launcher       " -ForegroundColor White
    Write-Host "  ============================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "   [1] ASL Model API     " -NoNewline; Write-Host "(FastAPI, port 8001)" -ForegroundColor Yellow
    Write-Host "   [2] LLM Backend       " -NoNewline; Write-Host "(Flask, port 5000)" -ForegroundColor Yellow
    Write-Host "   [3] Pocket TTS Server " -NoNewline; Write-Host "(FastAPI, port 8002)" -ForegroundColor Yellow
    Write-Host "   [4] Frontend          " -NoNewline; Write-Host "(Next.js, port 3000)" -ForegroundColor Yellow
    Write-Host "   [5] Auth Backend      " -NoNewline; Write-Host "(Flask, port 8000)" -ForegroundColor Yellow
    Write-Host "   [6] Start ALL Services" -NoNewline; Write-Host "(Each in a new window)" -ForegroundColor Green
    Write-Host "   [7] Exit" -ForegroundColor Red
    Write-Host ""
    Write-Host "  ============================================" -ForegroundColor Cyan
    Write-Host ""
}

Write-Banner

# Get user selection
$choice = Read-Host "   Select a service to start [1-7]"
$choice = $choice.Trim()

if ($choice -eq "7" -or $choice -eq "") {
    Write-Host "   Exiting launcher. Goodbye!" -ForegroundColor Gray
    Exit
}

if ($choice -eq "6") {
    Write-Host "   Spawning all services in separate windows..." -ForegroundColor Green
    Start-Process powershell -ArgumentList "-Command", "Set-Location '$Root'; .\start_all.ps1"
    Exit
}

$services = @{
    "1" = @{ "Name" = "ASL Model API"; "Script" = "start_asl_model.ps1" }
    "2" = @{ "Name" = "LLM Backend"; "Script" = "start_llm_backend.ps1" }
    "3" = @{ "Name" = "Pocket TTS Server"; "Script" = "start_pocket_tts.ps1" }
    "4" = @{ "Name" = "Frontend"; "Script" = "start_frontend.ps1" }
    "5" = @{ "Name" = "Auth Backend"; "Script" = "start_auth_backend.ps1" }
}

if ($services.ContainsKey($choice)) {
    $service = $services[$choice]
    $sName = $service.Name
    $sScript = $service.Script
    
    Write-Host ""
    Write-Host "   How would you like to run $sName?" -ForegroundColor Cyan
    Write-Host "   [1] Run in this window (blocking, easy to read logs)" -ForegroundColor Yellow
    Write-Host "   [2] Run in a new separate window (non-blocking)" -ForegroundColor Green
    $runType = Read-Host "   Select launch type [1-2]"
    $runType = $runType.Trim()
    
    if ($runType -eq "2") {
        Write-Host "   Spawning $sName in a new PowerShell window..." -ForegroundColor Green
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$Root'; .\$sScript" -WindowStyle Normal
    } else {
        Write-Host "   Starting $sName in current terminal. Press Ctrl+C to stop." -ForegroundColor Yellow
        Set-Location $Root
        & ".\$sScript"
    }
} else {
    Write-Host "   Invalid selection. Please run the script again." -ForegroundColor Red
}
