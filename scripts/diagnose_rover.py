import os
import sys
import time
import paramiko

ROVER_IP = "10.34.19.247"
ROVER_USER = "root"
ROVER_PASS = "0000"
REMOTE_DIR = "/root/novacare"

def main():
    print(f"Connecting to Rover at {ROVER_IP} for diagnostics...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect(ROVER_IP, username=ROVER_USER, password=ROVER_PASS, timeout=10)
        print("[OK] SSH connection successful.")
    except Exception as e:
        print(f"[ERROR] Failed to connect: {e}")
        sys.exit(1)

    # 1. Check running python processes
    print("\n--- Running Python Processes ---")
    stdin, stdout, stderr = ssh.exec_command("ps aux | grep python3")
    print(stdout.read().decode())

    # 2. Check listening ports
    print("--- Listening Ports (9000, 9999) ---")
    stdin, stdout, stderr = ssh.exec_command("netstat -tuln | grep -E '9000|9999' || ss -tuln | grep -E '9000|9999'")
    print(stdout.read().decode())

    # 3. Check logs if directories exist
    print("--- Check recent logs ---")
    stdin, stdout, stderr = ssh.exec_command("ls -la /root/novacare")
    print(stdout.read().decode())

    # 4. Proactively launch services if not running
    print("\n--- Proactively launching/restarting services ---")
    
    # Kill any existing ones to avoid port conflicts
    print("Killing any stale robot service or orchestrator processes...")
    ssh.exec_command("pkill -f robot_service.py")
    ssh.exec_command("pkill -f runtime_orchestrator")
    time.sleep(2)
    
    print("Starting Robot Service HAL (Flask on Port 9000)...")
    # We must set PYTHONPATH so imports work correctly
    cmd_robot = f"export PYTHONPATH={REMOTE_DIR} && cd {REMOTE_DIR} && nohup python3 services/robot/robot_service.py > robot_service.log 2>&1 &"
    ssh.exec_command(cmd_robot)
    time.sleep(2)

    print("Starting Central Runtime Orchestrator (Websocket on Port 9999)...")
    cmd_orchestrator = f"export PYTHONPATH={REMOTE_DIR} && cd {REMOTE_DIR} && nohup python3 -m optimized_runtime.orchestrator.runtime_orchestrator > orchestrator.log 2>&1 &"
    ssh.exec_command(cmd_orchestrator)
    time.sleep(3)

    # 5. Check if they launched successfully
    print("\n--- Verification after launch ---")
    stdin, stdout, stderr = ssh.exec_command("ps aux | grep python3")
    print(stdout.read().decode())

    stdin, stdout, stderr = ssh.exec_command("netstat -tuln | grep -E '9000|9999'")
    net_out = stdout.read().decode()
    print("Active Ports:")
    print(net_out)

    # 6. Read log files to catch immediate startup tracebacks
    print("--- Robot Service Log Snapshot ---")
    stdin, stdout, stderr = ssh.exec_command(f"tail -n 30 {REMOTE_DIR}/robot_service.log")
    print(stdout.read().decode())

    print("--- Orchestrator Log Snapshot ---")
    stdin, stdout, stderr = ssh.exec_command(f"tail -n 30 {REMOTE_DIR}/orchestrator.log")
    print(stdout.read().decode())

    ssh.close()

if __name__ == "__main__":
    main()
