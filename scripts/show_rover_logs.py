import paramiko
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

    print("=== ROBOT SERVICE (PORT 9000) LOGS ===")
    stdin, stdout, stderr = ssh.exec_command(f"tail -n 20 {REMOTE_DIR}/robot_service.log")
    safe_print(stdout.read().decode('utf-8', errors='replace'))

    print("\n=== CENTRAL ORCHESTRATOR (PORT 9999) LOGS ===")
    stdin, stdout, stderr = ssh.exec_command(f"tail -n 20 {REMOTE_DIR}/orchestrator.log")
    safe_print(stdout.read().decode('utf-8', errors='replace'))

    ssh.close()

if __name__ == "__main__":
    main()
