import paramiko
import sys
import time

from shared.configs.robot_config import ROBOT_IP as ROVER_IP
ROVER_USER = "root"
ROVER_PASS = "0000"
REMOTE_DIR = "/root/novacare"

def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ROVER_IP, username=ROVER_USER, password=ROVER_PASS)

    print("Running orchestrator synchronously to capture startup output/traceback...")
    cmd = f"export PYTHONPATH={REMOTE_DIR} && cd {REMOTE_DIR} && python3 -m optimized_runtime.orchestrator.runtime_orchestrator"
    
    stdin, stdout, stderr = ssh.exec_command(cmd)
    
    # Wait up to 5 seconds for it to start/crash
    time.sleep(5)
    
    # Read anything from stdout and stderr
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    
    print("--- STDOUT ---")
    print(out)
    print("--- STDERR ---")
    print(err)
    
    ssh.close()

if __name__ == "__main__":
    main()
