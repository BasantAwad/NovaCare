import paramiko
import sys
import time

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

    print("Running launcher synchronously to capture startup output...")
    cmd = f"export PYTHONPATH={REMOTE_DIR} && cd {REMOTE_DIR} && python3 -m optimized_runtime.runtime.launcher --mode serbot"
    
    stdin, stdout, stderr = ssh.exec_command(cmd)
    
    time.sleep(5)
    
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    
    print("--- STDOUT ---")
    safe_print(out)
    print("--- STDERR ---")
    safe_print(err)
    
    ssh.close()

if __name__ == "__main__":
    main()
