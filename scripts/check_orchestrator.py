import paramiko
import sys

from shared.configs.robot_config import ROBOT_IP as ROVER_IP
ROVER_USER = "root"
ROVER_PASS = "0000"

def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ROVER_IP, username=ROVER_USER, password=ROVER_PASS)

    print("=== ORCHESTRATOR NETSTAT PORT 9999 ===")
    stdin, stdout, stderr = ssh.exec_command("netstat -tulnp | grep :9999")
    print(stdout.read().decode('utf-8'))

    print("=== ORCHESTRATOR PROCESS ===")
    stdin, stdout, stderr = ssh.exec_command("ps aux | grep launcher | grep -v grep")
    print(stdout.read().decode('utf-8'))

    ssh.close()

if __name__ == "__main__":
    main()
