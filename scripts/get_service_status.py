import paramiko
import sys

from shared.configs.robot_config import ROBOT_IP as ROVER_IP
ROVER_USER = "root"
ROVER_PASS = "0000"

def safe_print(title, content):
    print(f"=== {title} ===")
    sys.stdout.flush()
    sys.stdout.buffer.write(content)
    print("\n")
    sys.stdout.flush()

def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ROVER_IP, username=ROVER_USER, password=ROVER_PASS)

    # 1. Active listening ports
    stdin, stdout, stderr = ssh.exec_command("netstat -tulnp")
    safe_print("NETSTAT ACTIVE PORTS", stdout.read())

    # 2. Log contents
    stdin, stdout, stderr = ssh.exec_command("tail -n 30 /root/novacare/robot_service.log")
    safe_print("ROBOT SERVICE LOG", stdout.read())

    stdin, stdout, stderr = ssh.exec_command("tail -n 30 /root/novacare/orchestrator.log")
    safe_print("ORCHESTRATOR LOG", stdout.read())

    ssh.close()

if __name__ == "__main__":
    main()
