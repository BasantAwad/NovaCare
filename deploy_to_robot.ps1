<#
deploy_to_robot.ps1
PowerShell deployment helper for copying project folders to a robot over SSH,
setting up a Python virtualenv, installing dependencies, and optionally starting services.

Usage example:
  .\deploy_to_robot.ps1 -User root -Host 10.34.19.247 -RemoteDir ~/novacare -StartServices -SudoInstall

Requires: scp and ssh available in PATH (Windows OpenSSH or Git for Windows).
#>

param(
    [Parameter(Mandatory=$true)][string]$User,
    [Parameter(Mandatory=$true)][string]$RemoteHost,
    [string]$RemoteDir = "~/novacare",
    [switch]$StartServices,
    [switch]$SudoInstall,
    [switch]$SkipPip,
    [string[]]$Include
)

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition

# Default include list
$includes = @('optimized_runtime','services/robot','shared','scripts')
if ($Include) { $includes += $Include }

$excludes = @('.git','apps/mobile','node_modules','dist','build','__pycache__','.venv','.pytest_cache')

function Check-Command($name) {
    $cmd = Get-Command $name -ErrorAction SilentlyContinue
    if (-not $cmd) {
        Write-Error "Required command '$name' not found in PATH. Please install OpenSSH (scp/ssh) or use WSL/Git Bash."
        exit 1
    }
}

Check-Command scp
Check-Command ssh

Write-Host ("Deploying to {0}@{1}:{2}" -f $User, $RemoteHost, $RemoteDir)

# Ensure remote directory exists
ssh $User@$RemoteHost "mkdir -p $RemoteDir"

# Copy include directories
foreach ($src in $includes) {
    $localPath = Join-Path $ScriptRoot $src
    if (Test-Path $localPath) {
        Write-Host "Copying $src..."
        $remoteTarget = "{0}@{1}:{2}/" -f $User, $RemoteHost, $RemoteDir
        & scp -r $localPath $remoteTarget
    }
    else {
        Write-Warning "$src not found, skipping"
    }
}

# Copy requirements files if present
$reqFiles = @('services/robot/requirements.txt','optimized_runtime/requirements.txt','requirements.txt')
foreach ($req in $reqFiles) {
    $localReq = Join-Path $ScriptRoot $req
    if (Test-Path $localReq) {
        Write-Host "Copying $req..."
        $remoteTarget = "{0}@{1}:{2}/" -f $User, $RemoteHost, $RemoteDir
        & scp $localReq $remoteTarget
    }
}

# Build remote setup commands
# Create a temporary local file with the remote setup script and SCP it to the robot
$remoteScript = @'
#!/usr/bin/env bash
set -e
cd "{REMOTE_DIR}"
# Create or update python venv
python3 -m venv venv || true
. venv/bin/activate
pip install --upgrade pip setuptools wheel
if [ "{SKIP_PIP}" != "true" ]; then
    if [ -f services/robot/requirements.txt ]; then
        pip install -r services/robot/requirements.txt || true
    fi
    if [ -f optimized_runtime/requirements.txt ]; then
        pip install -r optimized_runtime/requirements.txt || true
    fi
fi
'@

$remoteScript = $remoteScript -replace '\{REMOTE_DIR\}',$RemoteDir
$remoteScript = $remoteScript -replace '\{SKIP_PIP\}',$SkipPip.ToString().ToLower()

$tmpLocal = Join-Path $env:TEMP "novacare_setup.sh"
Set-Content -Path $tmpLocal -Value $remoteScript -Encoding UTF8

Write-Host ("Uploading remote setup script to {0}@{1}:{2}/novacare_setup.sh" -f $User, $RemoteHost, $RemoteDir)
$remoteTarget = "{0}@{1}:{2}/novacare_setup.sh" -f $User, $RemoteHost, $RemoteDir
& scp $tmpLocal $remoteTarget

Write-Host "Running remote setup script (this may take a while)..."
& ssh $User@$RemoteHost "bash $RemoteDir/novacare_setup.sh"

# Optionally run apt-get for camera/audio deps (requires sudo)
if ($SUDO_INSTALL) {
    Write-Host "Running remote apt-get install for camera/audio dependencies (requires sudo)..."
    & ssh $User@$RemoteHost "sudo apt-get update && sudo apt-get install -y gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good libopencv-dev libatlas-base-dev mpg123 || true"
}

# Optionally start services
if ($START_SERVICES) {
    Write-Host "Starting services on robot (robot_service + runtime launcher)..."
    & ssh $User@$RemoteHost "bash -lc 'cd $RemoteDir; . venv/bin/activate; nohup python3 services/robot/robot_service.py > robot_service.log 2>&1 &'"
    & ssh $User@$RemoteHost "bash -lc 'cd $RemoteDir; . venv/bin/activate; export RUNTIME_PORT=9999; nohup python3 -m optimized_runtime.runtime.launcher --mode serbot > runtime.log 2>&1 &'"
    Write-Host "Services started (logs: robot_service.log, runtime.log in $RemoteDir)"
}

Write-Host ("Deployment complete.`nImportant next steps on the robot:`n - Ensure the board-specific 'pop' library and LiDAR drivers are installed.`n - If using camera/GStreamer, verify GStreamer pipelines.`n - Check logs: {0}/robot_service.log and {0}/runtime.log" -f $RemoteDir)
