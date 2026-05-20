import asyncio
import websockets
import sys

async def test_connect():
    url = "ws://10.34.19.247:9999"
    print(f"Attempting to connect to WebSocket server at {url} from this machine...")
    try:
        async with websockets.connect(url, timeout=5) as websocket:
            print("[OK] Connected successfully!")
            # Send a test ping or request
            print("Sending test request...")
            await websocket.send('{"type": "ping"}')
            response = await websocket.recv()
            print(f"Received response: {response}")
    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_connect())
