import paramiko
import time
import sys

ROVER_IP = "10.34.19.247"
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

    print("1. Force freeing ports 9000 and 9999...")
    ssh.exec_command("fuser -k 9000/tcp")
    ssh.exec_command("fuser -k 9999/tcp")
    ssh.exec_command("pkill -f optimized_runtime")
    ssh.exec_command("pkill -f robot_service")
    time.sleep(3)

    print("2. Launching Robot Service (HAL Flask API on Port 9000)...")
    cmd_robot = f"export PYTHONPATH={REMOTE_DIR} && cd {REMOTE_DIR} && nohup python3 robot/robot_service.py > {REMOTE_DIR}/robot_service.log 2>&1 & disown"
    ssh.exec_command(cmd_robot)

    print("3. Launching Central Runtime Orchestrator (Websocket Server on Port 9999)...")
    cmd_launcher = f"export PYTHONPATH={REMOTE_DIR} && cd {REMOTE_DIR} && nohup python3 -m optimized_runtime.runtime.launcher --mode serbot > {REMOTE_DIR}/orchestrator.log 2>&1 & disown"
    ssh.exec_command(cmd_launcher)
    
    print("Waiting 12 seconds for TensorFlow and services to fully initialize...")
    time.sleep(12)

    print("\n4. Verifying active listening ports (9000 & 9999):")
    stdin, stdout, stderr = ssh.exec_command("netstat -tuln | grep -E '9000|9999' || ss -tuln | grep -E '9000|9999'")
    safe_print(stdout.read().decode('utf-8', errors='replace'))

    print("5. Verifying running Python processes:")
    stdin, stdout, stderr = ssh.exec_command("ps aux | grep python3")
    safe_print(stdout.read().decode('utf-8', errors='replace'))

    print("6. Reading Robot Service Log Snapshot:")
    stdin, stdout, stderr = ssh.exec_command(f"tail -n 25 {REMOTE_DIR}/robot_service.log")
    safe_print(stdout.read().decode('utf-8', errors='replace'))

    print("7. Reading Orchestrator Log Snapshot:")
    stdin, stdout, stderr = ssh.exec_command(f"tail -n 25 {REMOTE_DIR}/orchestrator.log")
    safe_print(stdout.read().decode('utf-8', errors='replace'))

    ssh.close()

if __name__ == "__main__":
    main()
