import paramiko

ROVER_IP = "10.34.19.247"
ROVER_USER = "root"
ROVER_PASS = "0000"

def try_kiosk(display):
    print(f"\n=== LAUNCHING ON {display} ===")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ROVER_IP, username=ROVER_USER, password=ROVER_PASS)
    
    cmd = (
        f"export DISPLAY={display} && "
        "chromium-browser --no-sandbox --kiosk --noerrdialogs --disable-translate "
        "--no-first-run --fast --fast-start --disable-infobars "
        "--use-fake-ui-for-media-stream "
        "--unsafely-treat-insecure-origin-as-secure='http://localhost:9000' "
        "--disable-features=TranslateUI --disk-cache-dir=/dev/null "
        "'http://localhost:9000'"
    )
    
    stdin, stdout, stderr = ssh.exec_command(cmd)
    
    import time
    time.sleep(4)
    
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    
    print("STDOUT:")
    print(out)
    print("STDERR:")
    print(err)
    
    ssh.close()

def main():
    try_kiosk(":0")
    try_kiosk(":1001")

if __name__ == "__main__":
    main()
