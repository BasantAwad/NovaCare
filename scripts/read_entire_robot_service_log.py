import paramiko
import sys

from shared.configs.robot_config import ROBOT_IP as ROVER_IP
ROVER_USER = "root"
ROVER_PASS = "0000"

def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ROVER_IP, username=ROVER_USER, password=ROVER_PASS)

    print("=== ENTIRE ROBOT SERVICE LOG ===")
    stdin, stdout, stderr = ssh.exec_command("cat /root/novacare/robot_service.log")
    content = stdout.read()
    sys.stdout.buffer.write(content)
    print()

    ssh.close()

if __name__ == "__main__":
    main()
