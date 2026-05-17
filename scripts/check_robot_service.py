import paramiko
import sys

ROVER_IP = "10.34.19.247"
ROVER_USER = "root"
ROVER_PASS = "0000"

def safe_print(title, lines):
    print(f"=== {title} ===")
    for line in lines:
        sys.stdout.buffer.write(line.encode('utf-8', errors='replace'))
        print()
    print()

def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ROVER_IP, username=ROVER_USER, password=ROVER_PASS)

    # 1. Check if robot_service is running
    stdin, stdout, stderr = ssh.exec_command("ps aux | grep robot_service.py | grep -v grep")
    safe_print("ROBOT_SERVICE PROCESS", stdout.read().decode('utf-8').splitlines())

    # 2. Check netstat for port 9000
    stdin, stdout, stderr = ssh.exec_command("netstat -tulnp | grep :9000")
    safe_print("NETSTAT PORT 9000", stdout.read().decode('utf-8').splitlines())

    # 3. Check logs of robot_service for any errors during startup
    stdin, stdout, stderr = ssh.exec_command("tail -n 25 /root/novacare/robot_service.log")
    safe_print("ROBOT_SERVICE STARTUP LOG", stdout.read().decode('utf-8').splitlines())

    ssh.close()

if __name__ == "__main__":
    main()
