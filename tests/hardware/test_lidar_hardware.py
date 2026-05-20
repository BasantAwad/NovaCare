import time
import sys
import os

def check_serial_ports():
    print("[0] Checking serial ports...")
    ports = ["/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyACM0"]
    found = False
    for port in ports:
        if os.path.exists(port):
            print(f"    -> Found potential LiDAR port: {port}")
            found = True
            # Check if port is locked by another process
            try:
                # Basic check, might not catch everything
                os.access(port, os.R_OK | os.W_OK)
            except:
                pass
    if not found:
        print("    -> ⚠️ WARNING: No common LiDAR serial ports found (/dev/ttyUSB0, etc).")
        print("       Is the LiDAR physically plugged into the USB port?")

def test_lidar():
    print("=======================================")
    print("      SERBot LiDAR Hardware Test       ")
    print("=======================================")
    
    check_serial_ports()
    
    try:
        from pop import LiDAR
    except ImportError:
        print("❌ Error: 'pop' library not found. Are you running this on the robot?")
        sys.exit(1)

    print("[1] Attempting to initialize LiDAR.Rplidar()...")
    # Note: If a Segmentation Fault occurs here, the script will crash immediately
    # and return to the terminal. This means the driver/hardware has a critical failure.
    try:
        lidar = LiDAR.Rplidar()
    except Exception as e:
        print(f"❌ Failed to instantiate LiDAR: {e}")
        sys.exit(1)
        
    print("[2] Attempting to connect...")
    try:
        lidar.connect()
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        sys.exit(1)
        
    print("[3] Attempting to start motor...")
    print("    ⚠️ IF THE SCRIPT CRASHES (Segmentation Fault) IMMEDIATELY AFTER THIS LINE, IT MEANS:")
    print("       1. Another process (like a ROS node) is already using the LiDAR.")
    print("       2. The LiDAR USB cable is loose or not receiving enough power.")
    print("       3. The driver failed to open the serial port but didn't throw a proper Python error.")
    print("    -> Try running:  sudo lsof /dev/ttyUSB0   to see what is blocking it.")
    try:
        lidar.startMotor()
        print("✅ LiDAR connected and motor started successfully!")
    except Exception as e:
        print(f"❌ Failed to start motor: {e}")
        sys.exit(1)

    print("[4] Reading data for 5 seconds...")
    start_time = time.time()
    valid_reads = 0
    while time.time() - start_time < 5:
        try:
            vectors = lidar.getVectors()
            if vectors and len(vectors) > 0:
                valid_reads += 1
                if valid_reads == 1:
                    # Print the first successfully read point just to show it's working
                    if isinstance(vectors, dict):
                        k = list(vectors.keys())[0]
                        print(f"   -> Sample Data - Angle: {k}, Distance: {vectors[k]}mm")
                    else:
                        print(f"   -> Sample Data: {vectors[0]}")
            time.sleep(0.1)
        except Exception as e:
            print(f"❌ Error reading vectors: {e}")
            break

    print("=======================================")
    print(f"Test complete. Total successful scans: {valid_reads}")
    if valid_reads == 0:
        print("❌ No data received. Motor might be spinning but laser is off, or serial port is blocked.")
    else:
        print("✅ LiDAR Hardware is fully functional!")
    print("=======================================")
    
    try:
        lidar.stopMotor()
    except:
        pass

if __name__ == '__main__':
    test_lidar()
