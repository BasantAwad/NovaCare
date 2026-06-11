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
$MathTutorKDir = "DISABLED_TO_FIX_OPENSSL_BUG"
$MathTutorKVenv = Join-Path $MathTutorKDir ".tts-venv"
$MathTutorKPocketTts = Join-Path $MathTutorKVenv "Scripts\pocket-tts.exe"

Write-Host "=== NovaCare - Pocket TTS Server ===" -ForegroundColor Cyan

# 1. Check if we can reuse the Math-TutorK virtual environment
if (Test-Path $MathTutorKPocketTts) {
    Write-Host "[OK] Detected existing Pocket TTS venv in Math-TutorK!" -ForegroundColor Green
    Write-Host "[*] Reusing Math-TutorK's Pocket TTS installation to save space." -ForegroundColor Cyan
    Write-Host "[*] Starting Pocket TTS on http://localhost:8002..." -ForegroundColor Cyan
    $MathTutorKPython = Join-Path $MathTutorKVenv "Scripts\python.exe"
    & $MathTutorKPython -m pocket_tts serve --host 0.0.0.0 --port 8002
} else {
    Write-Host "[!] Math-TutorK's Pocket TTS venv not found at: $MathTutorKPocketTts" -ForegroundColor Yellow
    Write-Host "[*] Setting up local Pocket TTS service..." -ForegroundColor Cyan
    
    $UvBin = if ($env:UV_BIN) { $env:UV_BIN } else { "uv" }
    $HasUv = Get-Command $UvBin -ErrorAction SilentlyContinue
    
    $LocalPocketTts = Join-Path $LocalVenvDir "Scripts\pocket-tts.exe"

    # Create local venv if it doesn't exist or is invalid
    if (-not (Test-Path $LocalPocketTts)) {
        if (Test-Path $LocalVenvDir) {
            Write-Host "[*] Cleaning old invalid venv..." -ForegroundColor Yellow
            Remove-Item -Recurse -Force $LocalVenvDir -ErrorAction SilentlyContinue
        }
        if ($HasUv) {
            Write-Host "[*] Creating virtual environment with uv..." -ForegroundColor Cyan
            & $UvBin venv $LocalVenvDir --python 3.10
        } else {
            # Try to use python 3.10 or 3.12 to avoid python 3.13 PyTorch wheels issue
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
                & py -3.10 -m venv $LocalVenvDir
            } elseif ($hasPy12) {
                Write-Host "[*] Creating virtual environment with Python 3.12..." -ForegroundColor Cyan
                & py -3.12 -m venv $LocalVenvDir
            } else {
                Write-Host "[*] Creating virtual environment with default python..." -ForegroundColor Cyan
                python -m venv $LocalVenvDir
            }
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
    & $LocalPocketTts serve --host 0.0.0.0 --port 8002
}
