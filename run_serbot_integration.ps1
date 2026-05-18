#requires -Version 5.1

param(
  [Parameter(Mandatory=$false)]
  [string]$RobotIp = $null,
  [Parameter(Mandatory=$false)]
  [string]$RobotUser = "root",
  [Parameter(Mandatory=$false)]
  [object]$DeployOnly = $false,
  [Parameter(Mandatory=$false)]
  [int]$SshPort = 22
)


$ErrorActionPreference = "Stop"

# Allow environment variable `ROBOT_IP` to override default when provided,
# otherwise use the previous default IP as a fallback.
if (-not $RobotIp -or $RobotIp -eq "") {
  if ($env:ROBOT_IP) { $RobotIp = $env:ROBOT_IP } else { $RobotIp = "192.168.137.206" }
}

# Coerce DeployOnly to a real boolean even if PowerShell passed it as a string
if ($DeployOnly -is [bool]) {
  $DeployOnlyBool = [bool]$DeployOnly
} else {
  $DeployOnlyBool = [System.Convert]::ToBoolean($DeployOnly.ToString())
}

function Write-Banner {
  param([string]$Text)
  Write-Host ""
  Write-Host "============================================================" -ForegroundColor Cyan
  Write-Host $Text -ForegroundColor White
  Write-Host "============================================================" -ForegroundColor Cyan
}

function Run-Command {
  param([string]$Cmd)
  Write-Host "[CMD] $Cmd" -ForegroundColor Yellow
  Invoke-Expression $Cmd
}

$RepoRoot = (Get-Item -Path ".").FullName

Write-Banner "NovaCare SERBot Integration Runner"
Write-Host "Repo root: $RepoRoot" -ForegroundColor Gray
Write-Host "Robot: $RobotUser@$RobotIp" -ForegroundColor Gray
Write-Host "DeployOnly: $DeployOnlyBool" -ForegroundColor Gray

if (-not $DeployOnlyBool) {
  Write-Banner "Starting local services (ASL 8000, LLM 5000, Robot 9000)"

  # ASL
  $aslDir = Join-Path $RepoRoot "services\asl"
  if (!(Test-Path (Join-Path $aslDir "venv"))) {
    Write-Host "ASL venv not found at $aslDir\venv - creating..." -ForegroundColor Yellow
    $venvPath = Join-Path $aslDir "venv"
    try {
      python -m venv $venvPath
    } catch {
      try { py -3 -m venv $venvPath } catch { throw "Could not create virtualenv - ensure Python 3 is installed and on PATH." }
    }
    $pipExe = Join-Path $venvPath "Scripts\pip.exe"
    if (Test-Path (Join-Path $aslDir "requirements.txt")) {
      if (Test-Path $pipExe) { & $pipExe install -r (Join-Path $aslDir "requirements.txt") } else { & (Join-Path $venvPath "Scripts\python.exe") -m pip install -r (Join-Path $aslDir "requirements.txt") }
    } else {
      Write-Host "No requirements.txt in $aslDir - skipping pip install" -ForegroundColor Yellow
    }
  }
  Start-Process powershell -NoNewWindow -ArgumentList '-Command', ("cd `"$aslDir`"; & .\venv\Scripts\Activate.ps1; python -m api.main --port 8000")

  # LLM
  $llmDir = Join-Path $RepoRoot "services\llm"
  if (!(Test-Path (Join-Path $llmDir "venv"))) {
    Write-Host "LLM venv not found at $llmDir\venv - creating..." -ForegroundColor Yellow
    $venvPath = Join-Path $llmDir "venv"
    try {
      python -m venv $venvPath
    } catch {
      try { py -3 -m venv $venvPath } catch { throw "Could not create virtualenv - ensure Python 3 is installed and on PATH." }
    }
    $pipExe = Join-Path $venvPath "Scripts\pip.exe"
    if (Test-Path (Join-Path $llmDir "requirements.txt")) {
      if (Test-Path $pipExe) { & $pipExe install -r (Join-Path $llmDir "requirements.txt") } else { & (Join-Path $venvPath "Scripts\python.exe") -m pip install -r (Join-Path $llmDir "requirements.txt") }
    } else {
      Write-Host "No requirements.txt in $llmDir - skipping pip install" -ForegroundColor Yellow
    }
  }
  Start-Process powershell -NoNewWindow -ArgumentList '-Command', ("cd `"$llmDir`"; & .\venv\Scripts\Activate.ps1; python start_server.py")

  # Robot service (HAL REST API)
  $robotDir = Join-Path $RepoRoot "services\robot"
  if (!(Test-Path (Join-Path $robotDir "venv"))) {
    Write-Host "Robot venv not found at $robotDir\venv - creating..." -ForegroundColor Yellow
    $venvPath = Join-Path $robotDir "venv"
    try {
      python -m venv $venvPath
    } catch {
      try { py -3 -m venv $venvPath } catch { throw "Could not create virtualenv - ensure Python 3 is installed and on PATH." }
    }
    $pipExe = Join-Path $venvPath "Scripts\pip.exe"
    if (Test-Path (Join-Path $robotDir "requirements.txt")) {
      if (Test-Path $pipExe) { & $pipExe install -r (Join-Path $robotDir "requirements.txt") } else { & (Join-Path $venvPath "Scripts\python.exe") -m pip install -r (Join-Path $robotDir "requirements.txt") }
    } else {
      Write-Host "No requirements.txt in $robotDir - skipping pip install" -ForegroundColor Yellow
    }
  }
  Start-Process powershell -NoNewWindow -ArgumentList '-Command', ("cd `"$robotDir`"; & .\venv\Scripts\Activate.ps1; python robot_service.py")

  Write-Host "Waiting for local services to boot..." -ForegroundColor Gray
  Start-Sleep -Seconds 8
}

Write-Banner "Deploy optimized_runtime to SERBot"
Run-Command "bash optimized_runtime/scripts/deploy_serbot.sh $RobotIp $RobotUser"

Write-Host "Waiting for robot runtime/WebSocket..." -ForegroundColor Gray
Start-Sleep -Seconds 6

Write-Banner "Run health checks (from your PC)"
Run-Command "bash optimized_runtime/scripts/health_check.sh $RobotIp"

Write-Host "Done." -ForegroundColor Green

