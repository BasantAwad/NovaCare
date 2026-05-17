#requires -Version 5.1

param(
  [Parameter(Mandatory=$false)]
  [string]$RobotIp = "10.34.19.247",
  [Parameter(Mandatory=$false)]
  [string]$RobotUser = "root",
  [Parameter(Mandatory=$false)]
  [object]$DeployOnly = $false,
  [Parameter(Mandatory=$false)]
  [int]$SshPort = 22
)


$ErrorActionPreference = "Stop"

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
    throw "ASL venv not found at $aslDir\venv. Create it and install requirements.txt first."
  }
  Start-Process powershell -NoNewWindow -ArgumentList '-Command', ("cd `"$aslDir`"; & .\venv\Scripts\Activate.ps1; python -m api.main --port 8000")

  # LLM
  $llmDir = Join-Path $RepoRoot "services\llm"
  if (!(Test-Path (Join-Path $llmDir "venv"))) {
    throw "LLM venv not found at $llmDir\venv. Create it and install requirements.txt first."
  }
  Start-Process powershell -NoNewWindow -ArgumentList '-Command', ("cd `"$llmDir`"; & .\venv\Scripts\Activate.ps1; python start_server.py")

  # Robot service (HAL REST API)
  $robotDir = Join-Path $RepoRoot "services\robot"
  if (!(Test-Path (Join-Path $robotDir "venv"))) {
    throw "Robot venv not found at $robotDir\venv. Create it and install requirements.txt first."
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

