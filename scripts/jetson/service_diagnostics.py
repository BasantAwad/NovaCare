import requests
import time
import subprocess
import os
import sys

# Configuration
SERVICES = {
    "Robot Hardware": {"url": "http://localhost:9000/health", "port": 9000},
    "ASL Recognition": {"url": "http://localhost:8001/health", "port": 8001},
    "LLM Chat": {"url": "http://localhost:5000/", "port": 5000},
    "Frontend": {"url": "http://localhost:3000", "port": 3000}
}

def print_header(text):
    print("\n" + "="*50)
    print(f"  {text}")
    print("="*50)

def check_health(name, config):
    print(f"[*] Checking {name} health on port {config['port']}...")
    try:
        response = requests.get(config['url'], timeout=5)
        if response.status_code == 200:
            print(f"✅ {name} is UP and healthy!")
            try:
                data = response.json()
                print(f"    Status Details: {data}")
            except:
                print("    (Response is not JSON but returned 200 OK)")
            return True
        else:
            print(f"⚠️ {name} returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ {name} is DOWN (Connection Refused).")
        return False
    except Exception as e:
        print(f"❌ Error checking {name}: {e}")
        return False

def test_asl_mock():
    print_header("Testing ASL Recognition Inference")
    # A tiny 1x1 black pixel base64 to test API response without heavy data
    mock_frame = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    url = "http://localhost:8001/predict"
    
    print("[*] Sending mock frame to ASL service...")
    try:
        res = requests.post(url, json={"image": mock_frame}, timeout=10)
        if res.status_code == 200:
            print("✅ ASL API responded successfully!")
            print(f"    Result: {res.json()}")
        else:
            print(f"❌ ASL API failed with code {res.status_code}: {res.text}")
    except Exception as e:
        print(f"❌ ASL API test failed: {e}")

def test_robot_camera():
    print_header("Testing Robot Camera Stream")
    url = "http://localhost:9000/api/camera/frame"
    print("[*] Requesting single frame from Robot Service...")
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            print("✅ Robot Camera Service is functional!")
            print("    Successfully retrieved base64 frame.")
        else:
            print(f"❌ Robot Camera failed: {res.text}")
    except Exception as e:
        print(f"❌ Robot Camera test failed: {e}")

def main_menu():
    while True:
        print_header("NovaCare Jetson Diagnostic Tool")
        print("1. Run Health Check for ALL services")
        print("2. Test ASL Recognition (Mock Prediction)")
        print("3. Test Robot Camera (Capture Frame)")
        print("4. Check Port Occupancy (lsof)")
        print("5. Exit")
        
        choice = input("\nSelect an option [1-5]: ")
        
        if choice == "1":
            for name, config in SERVICES.items():
                check_health(name, config)
        elif choice == "2":
            test_asl_mock()
        elif choice == "3":
            test_robot_camera()
        elif choice == "4":
            print_header("Current Port Usage")
            os.system("lsof -iTCP -sTCP:LISTEN")
        elif choice == "5":
            print("Exiting diagnostics.")
            break
        else:
            print("Invalid choice, try again.")
        
        input("\nPress Enter to return to menu...")

if __name__ == "__main__":
    main_menu()
