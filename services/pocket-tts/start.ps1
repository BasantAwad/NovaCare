# Requires -Version 5.1
<#
.SYNOPSIS
    NovaCare - Pocket TTS Service Launcher
.DESCRIPTION
    Launches the Pocket TTS server on port 8002.
    It will first check if the Math-TutorK project's virtual environment exists
    to reuse the downloaded model and dependencies to save disk space and time.
    Otherwise, it creates a local venv and installs pocket-tts.
#>

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LocalVenvDir = Join-Path $ScriptDir ".venv"

# Path to Math-TutorK project's virtual environment
$MathTutorKDir = "G:\OneDrive - Alamein International University\Uni stuff\semester 8 - Spring 25-26\AIE314 AI-Based Programming\project\main project\Math-TutorK"
$MathTutorKVenv = Join-Path $MathTutorKDir ".tts-venv"
$MathTutorKPocketTts = Join-Path $MathTutorKVenv "Scripts\pocket-tts.exe"

Write-Host "=== NovaCare - Pocket TTS Server ===" -ForegroundColor Cyan

# 1. Check if we can reuse the Math-TutorK virtual environment
if (Test-Path $MathTutorKPocketTts) {
    Write-Host "[OK] Detected existing Pocket TTS venv in Math-TutorK!" -ForegroundColor Green
    Write-Host "[*] Reusing Math-TutorK's Pocket TTS installation to save space." -ForegroundColor Cyan
    Write-Host "[*] Starting Pocket TTS on http://localhost:8002..." -ForegroundColor Cyan
    & $MathTutorKPocketTts serve --host 0.0.0.0 --port 8002 --voice alba
} else {
    Write-Host "[!] Math-TutorK's Pocket TTS venv not found at: $MathTutorKPocketTts" -ForegroundColor Yellow
    Write-Host "[*] Setting up local Pocket TTS service..." -ForegroundColor Cyan
    
    $UvBin = if ($env:UV_BIN) { $env:UV_BIN } else { "uv" }
    $HasUv = Get-Command $UvBin -ErrorAction SilentlyContinue
    
    # Create local venv if it doesn't exist
    if (-not (Test-Path $LocalVenvDir)) {
        if ($HasUv) {
            Write-Host "[*] Creating virtual environment with uv..." -ForegroundColor Cyan
            & $UvBin venv $LocalVenvDir --python 3.12
        } else {
            Write-Host "[*] Creating virtual environment with python..." -ForegroundColor Cyan
            python -m venv $LocalVenvDir
        }
    }
    
    $LocalPocketTts = Join-Path $LocalVenvDir "Scripts\pocket-tts.exe"
    
    # Install pocket-tts if not installed
    if (-not (Test-Path $LocalPocketTts)) {
        Write-Host "[*] Installing pocket-tts into virtual environment..." -ForegroundColor Cyan
        if ($HasUv) {
            $PythonBin = Join-Path $LocalVenvDir "Scripts\python.exe"
            & $UvBin pip install pocket-tts --python $PythonBin
        } else {
            $PipBin = Join-Path $LocalVenvDir "Scripts\pip.exe"
            & $PipBin install pocket-tts
        }
    }
    
    Write-Host "[OK] Local Pocket TTS ready!" -ForegroundColor Green
    Write-Host "[*] Starting Pocket TTS on http://localhost:8002..." -ForegroundColor Cyan
    & $LocalPocketTts serve --host 0.0.0.0 --port 8002 --voice alba
}
