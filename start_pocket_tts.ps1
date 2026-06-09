$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$WorkDir = Join-Path $Root "services\pocket-tts"
Set-Location $WorkDir

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "   NovaCare - Starting Pocket TTS Server" -ForegroundColor White
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

.\start.ps1
