import paramiko
import time
import sys

from shared.configs.robot_config import ROBOT_IP as ROVER_IP
ROVER_USER = "root"
ROVER_PASS = "0000"
REMOTE_DIR = "/root/novacare"

def safe_print(text):
    sys.stdout.buffer.write(text.encode('utf-8', errors='replace'))
    print()

def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ROVER_IP, username=ROVER_USER, password=ROVER_PASS)

    print("1. Killing any existing rover processes...")
    ssh.exec_command("pkill -f robot_service.py")
    ssh.exec_command("pkill -f launcher.py")
    ssh.exec_command("pkill -f runtime_orchestrator")
    time.sleep(2)

    print("2. Launching Robot Service (HAL Flask API on Port 9000)...")
    cmd_robot = f"export PYTHONPATH={REMOTE_DIR} && cd {REMOTE_DIR} && nohup python3 services/robot/robot_service.py > {REMOTE_DIR}/robot_service.log 2>&1 &"
    ssh.exec_command(cmd_robot)
    time.sleep(3)

    print("3. Launching Central Runtime Orchestrator (Websocket Server on Port 9999)...")
    cmd_launcher = f"export PYTHONPATH={REMOTE_DIR} && cd {REMOTE_DIR} && nohup python3 -m optimized_runtime.runtime.launcher --mode serbot > {REMOTE_DIR}/orchestrator.log 2>&1 &"
    ssh.exec_command(cmd_launcher)
    time.sleep(5)

    print("\n4. Verifying running Python processes:")
    stdin, stdout, stderr = ssh.exec_command("ps aux | grep python3")
    safe_print(stdout.read().decode('utf-8', errors='replace'))

    print("5. Verifying active listening ports (9000 & 9999):")
    stdin, stdout, stderr = ssh.exec_command("netstat -tuln | grep -E '9000|9999' || ss -tuln | grep -E '9000|9999'")
    safe_print(stdout.read().decode('utf-8', errors='replace'))

    print("6. Reading Robot Service Log Startup Snapshot:")
    stdin, stdout, stderr = ssh.exec_command(f"tail -n 20 {REMOTE_DIR}/robot_service.log")
    safe_print(stdout.read().decode('utf-8', errors='replace'))

    print("7. Reading Orchestrator Log Startup Snapshot:")
    stdin, stdout, stderr = ssh.exec_command(f"tail -n 20 {REMOTE_DIR}/orchestrator.log")
    safe_print(stdout.read().decode('utf-8', errors='replace'))

    ssh.close()

if __name__ == "__main__":
    main()
