import httpx
import sys

def main():
    url = "http://10.34.19.247:9000/health"
    print(f"Sending GET request to {url}...")
    try:
        r = httpx.get(url, timeout=5.0)
        print(f"Status Code: {r.status_code}")
        print(f"Response: {r.text}")
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    main()
