import paramiko

ROVER_IP = "10.34.19.247"
ROVER_USER = "root"
ROVER_PASS = "0000"

def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ROVER_IP, username=ROVER_USER, password=ROVER_PASS)

    print("=== ACTIVE PORTS ===")
    stdin, stdout, stderr = ssh.exec_command("netstat -tuln | grep -E '9000|9999' || ss -tuln | grep -E '9000|9999'")
    print(stdout.read().decode())
    
    ssh.close()

if __name__ == "__main__":
    main()
