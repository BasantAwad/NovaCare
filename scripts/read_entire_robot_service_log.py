import paramiko
import sys

ROVER_IP = "10.34.19.247"
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
