import paramiko
from shared.configs.robot_config import ROBOT_IP as ROVER_IP

ROVER_USER = "root"
ROVER_PASS = "0000"

def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ROVER_IP, username=ROVER_USER, password=ROVER_PASS)

    print("=== FIREWALL DIAGNOSTICS ===")
    stdin, stdout, stderr = ssh.exec_command("ufw status")
    ufw_out = stdout.read().decode().strip()
    print("UFW Status:")
    print(ufw_out)
    
    if "active" in ufw_out:
        print("UFW is active! Proactively allowing ports 9000 and 9999...")
        ssh.exec_command("ufw allow 9000/tcp")
        ssh.exec_command("ufw allow 9999/tcp")
        ssh.exec_command("ufw reload")
        
        stdin, stdout, stderr = ssh.exec_command("ufw status")
        print("Updated UFW Status:")
        print(stdout.read().decode())
    else:
        print("UFW is inactive or not installed.")

    ssh.close()

if __name__ == "__main__":
    main()
