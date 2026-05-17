import os
import sys
import subprocess

def install_dependencies():
    print("Checking deployment dependencies...")
    try:
        import paramiko
        import scp
        print("[OK] paramiko and scp are already installed.")
    except ImportError:
        print("Installing paramiko and scp package via pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "paramiko", "scp"])
        print("[OK] Dependencies installed successfully.")

# Execute dependency check first
install_dependencies()

import paramiko
from scp import SCPClient

ROVER_IP = "10.34.19.247"
ROVER_USER = "root"
ROVER_PASS = "0000"
REMOTE_DIR = "/root/novacare"

def progress(filename, size, sent):
    sys.stdout.write(f"\rUploading {filename.decode('utf-8') if isinstance(filename, bytes) else filename}: {sent}/{size} bytes ({(sent/size)*100:.1f}%)")
    sys.stdout.flush()

def main():
    print(f"\n>>> Starting Deployment to Rover at {ROVER_IP}...")
    
    # 1. Connect SSH
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        print(f"Connecting to {ROVER_USER}@{ROVER_IP}...")
        ssh.connect(ROVER_IP, username=ROVER_USER, password=ROVER_PASS, timeout=10)
        print("[OK] SSH connection established successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to connect to rover via SSH: {e}")
        print("Please ensure the rover is turned on, connected to the same network, and that SSH is enabled.")
        sys.exit(1)
        
    # 2. Create remote directory structure
    print("Preparing remote directory...")
    ssh.exec_command(f"mkdir -p {REMOTE_DIR}/services/robot")
    ssh.exec_command(f"mkdir -p {REMOTE_DIR}/optimized_runtime")
    
    # Get workspace root directory path
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 3. SCP Transfer
    with SCPClient(ssh.get_transport(), progress=progress) as scp_client:
        # A. Copy requirements.txt
        print("\nUploading requirements.txt...")
        req_path = os.path.join(base_dir, "requirements.txt")
        scp_client.put(req_path, remote_path=f"{REMOTE_DIR}/requirements.txt")
        
        # B. Copy services/robot
        print("\nUploading services/robot...")
        robot_service_dir = os.path.join(base_dir, "services", "robot")
        for root, dirs, files in os.walk(robot_service_dir):
            # Skip virtual environments and caches
            if "venv" in root or "__pycache__" in root or ".git" in root:
                continue
                
            # Create remote subdirectory
            rel_path = os.path.relpath(root, robot_service_dir)
            remote_sub = f"{REMOTE_DIR}/services/robot"
            if rel_path != ".":
                remote_sub = os.path.join(remote_sub, rel_path).replace("\\", "/")
                ssh.exec_command(f"mkdir -p {remote_sub}")
                
            for file in files:
                if file.endswith(".pyc") or file == ".env":
                    continue
                local_file = os.path.join(root, file)
                remote_file = os.path.join(remote_sub, file).replace("\\", "/")
                scp_client.put(local_file, remote_path=remote_file)
                
        # C. Copy optimized_runtime
        print("\nUploading optimized_runtime...")
        runtime_dir = os.path.join(base_dir, "optimized_runtime")
        for root, dirs, files in os.walk(runtime_dir):
            if "__pycache__" in root or ".git" in root or "robot_ui" in root: # Skip UI if only backend Option B
                continue
                
            rel_path = os.path.relpath(root, runtime_dir)
            remote_sub = f"{REMOTE_DIR}/optimized_runtime"
            if rel_path != ".":
                remote_sub = os.path.join(remote_sub, rel_path).replace("\\", "/")
                ssh.exec_command(f"mkdir -p {remote_sub}")
                
            for file in files:
                if file.endswith(".pyc"):
                    continue
                local_file = os.path.join(root, file)
                remote_file = os.path.join(remote_sub, file).replace("\\", "/")
                scp_client.put(local_file, remote_path=remote_file)

    print("\n\n[OK] All files uploaded successfully!")
    
    # 4. Install remote dependencies
    print("\nPreparing remote package installation...")
    
    # Define version-agnostic libraries for maximum compatibility
    libs = [
        "flask", "flask-cors", "python-dotenv", 
        "gTTS", "SpeechRecognition", "websockets", 
        "httpx", "psutil", "pydantic"
    ]
    
    print(f"Installing version-agnostic libraries for older python environments: {', '.join(libs)}")
    stdin, stdout, stderr = ssh.exec_command(f"pip3 install {' '.join(libs)}")
    exit_status = stdout.channel.recv_exit_status()
    if exit_status == 0:
        print("[OK] All python dependencies installed successfully on the rover.")
    else:
        print(f"[WARN] Pip install returned exit code {exit_status}.")
        print("Error details:")
        print(stderr.read().decode())
        print("Note: If some optional libraries failed, you can run them manually on the rover.")
        
    print("\n*** DEPLOYMENT COMPLETE! ***")
    print(f"You can now run the services natively (Option B) by SSHing into the rover and running:")
    print(f"  cd {REMOTE_DIR}")
    print(f"  python3 services/robot/robot_service.py &")
    print(f"  python3 -m optimized_runtime.runtime.launcher --mode serbot")
    
    ssh.close()

if __name__ == "__main__":
    main()
