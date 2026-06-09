$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "   NovaCare - Starting ALL Services" -ForegroundColor White
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "This will spawn multiple new powershell windows." -ForegroundColor Yellow

Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$Root'; .\start_asl_model.ps1" -WindowStyle Normal
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$Root'; .\start_llm_backend.ps1" -WindowStyle Normal
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$Root'; .\start_pocket_tts.ps1" -WindowStyle Normal
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$Root'; .\start_frontend.ps1" -WindowStyle Normal
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$Root'; .\start_auth_backend.ps1" -WindowStyle Normal

Write-Host "All services have been launched in separate windows." -ForegroundColor Green
