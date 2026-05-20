import paramiko

ROVER_IP = "10.34.19.247"
ROVER_USER = "root"
ROVER_PASS = "0000"

def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ROVER_IP, username=ROVER_USER, password=ROVER_PASS)

    print("=== CHECKING RUNNING CHROMIUM PROCESSES ===")
    stdin, stdout, stderr = ssh.exec_command("ps aux | grep -i chromium")
    print(stdout.read().decode())

    print("=== CHECKING USER SODA DISPLAYS ===")
    stdin, stdout, stderr = ssh.exec_command("w")
    print(stdout.read().decode())

    print("=== SYSTEM DISPLAY PATHS ===")
    stdin, stdout, stderr = ssh.exec_command("ls -la /tmp/.X11-unix")
    print(stdout.read().decode())
    
    ssh.close()

if __name__ == "__main__":
    main()
