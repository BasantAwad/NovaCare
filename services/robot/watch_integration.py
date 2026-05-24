"""
Smart Watch Integration for NovaCare Robot
==========================================
Real-time heart rate and vital data from HRYFINE smartwatch via BLE.

Provides:
  - async watch client that connects to HRYFINE device
  - Continuous data monitoring and updates
  - Thread-safe global state for latest heart rate/battery
"""

import asyncio
import threading
import sys
import os
from datetime import datetime
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass, asdict

# Try to import watch protocol from sibling directory
watch_module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../watch/ble-test"))
if os.path.exists(watch_module_path):
    sys.path.insert(0, watch_module_path)

try:
    from watch_protocol import HRYFINEProtocol, WatchData
    from watch_client import HRYFINEWatchClient
    WATCH_AVAILABLE = True
except ImportError:
    print("⚠️  Watch module not available. Running in simulation mode.")
    WATCH_AVAILABLE = False


@dataclass
class RoverVitals:
    """Current rover vital signs from watch"""
    heart_rate: Optional[int] = None
    steps: Optional[int] = None
    battery: Optional[int] = None
    timestamp: datetime = None
    raw_data: Optional[bytes] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict"""
        return {
            'heart_rate': self.heart_rate,
            'steps': self.steps,
            'battery': self.battery,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
        }


class WatchIntegration:
    """
    Thread-safe watch integration manager
    
    Usage:
        manager = WatchIntegration("C2:FC:28:B7:1C:1B")
        manager.start()  # Start background monitoring
        # ... later ...
        vitals = manager.get_latest_vitals()
        manager.stop()
    """
    
    def __init__(
        self, 
        device_address: str,
        on_data_callback: Optional[Callable[[RoverVitals], None]] = None,
        simulation_mode: bool = False
    ):
        """
        Args:
            device_address: BLE address of HRYFINE watch (e.g., "C2:FC:28:B7:1C:1B")
            on_data_callback: Optional callback for each data update
            simulation_mode: If True, generate fake data for testing
        """
        self.device_address = device_address
        self.on_data_callback = on_data_callback
        self.simulation_mode = simulation_mode or not WATCH_AVAILABLE
        self.client: Optional[HRYFINEWatchClient] = None
        
        # Thread-safe state
        self._lock = threading.Lock()
        self._vitals = RoverVitals(timestamp=datetime.now())
        self._is_monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        # Simulation state
        self._sim_heart_rate = 72
    
    def _generate_fake_data(self) -> RoverVitals:
        """Generate realistic fake vital data for testing"""
        import random
        # Simulate realistic heart rate variation
        self._sim_heart_rate += random.randint(-3, 3)
        self._sim_heart_rate = max(60, min(100, self._sim_heart_rate))
        
        return RoverVitals(
            heart_rate=self._sim_heart_rate,
            steps=random.randint(500, 15000),
            battery=random.randint(20, 100),
            timestamp=datetime.now(),
        )
    
    async def _async_monitor_loop(self):
        """Async loop that continuously reads watch data"""
        if self.simulation_mode:
            print("🎭 Watch in SIMULATION mode (fake data)")
            # Simulate data in polling intervals
            while self._is_monitoring:
                vitals = self._generate_fake_data()
                with self._lock:
                    self._vitals = vitals
                
                if self.on_data_callback:
                    self.on_data_callback(vitals)
                
                await asyncio.sleep(2)  # Poll every 2 seconds
        else:
            if not self.client:
                self.client = HRYFINEWatchClient(self.device_address)
            
            # Connect
            if not await self.client.connect():
                print(f"❌ Failed to connect to watch {self.device_address}")
                return
            
            # Define callback
            def on_watch_data(data: WatchData):
                vitals = RoverVitals(
                    heart_rate=data.heart_rate,
                    steps=data.steps,
                    battery=data.battery,
                    timestamp=data.timestamp or datetime.now(),
                    raw_data=data.raw_message,
                )
                
                with self._lock:
                    self._vitals = vitals
                
                if self.on_data_callback:
                    self.on_data_callback(vitals)
            
            # Start monitoring
            try:
                await self.client.start_monitoring(on_watch_data)
            finally:
                await self.client.disconnect()
    
    def _run_monitor_thread(self):
        """Run the async monitor loop in a dedicated thread"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._async_monitor_loop())
        except Exception as e:
            print(f"❌ Monitor thread error: {e}")
        finally:
            with self._lock:
                self._is_monitoring = False
    
    def start(self):
        """Start background watch monitoring"""
        if self._is_monitoring:
            print("⚠️  Watch monitoring already running")
            return
        
        with self._lock:
            self._is_monitoring = True
        
        mode_text = "SIMULATION" if self.simulation_mode else f"{self.device_address}"
        print(f"📱 Starting watch monitoring ({mode_text})...")
        
        self._monitor_thread = threading.Thread(
            target=self._run_monitor_thread,
            daemon=True,
            name="WatchMonitor"
        )
        self._monitor_thread.start()
        print("✅ Watch monitor thread started")
    
    def stop(self):
        """Stop background watch monitoring"""
        with self._lock:
            self._is_monitoring = False
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        
        if self.client:
            # Can't await here, so disconnect synchronously if possible
            asyncio.run(self.client.disconnect())
        
        print("🛑 Watch monitoring stopped")
    
    def get_latest_vitals(self) -> RoverVitals:
        """Get the most recent vital signs (thread-safe)"""
        with self._lock:
            return self._vitals
    
    def is_connected(self) -> bool:
        """Check if watch is currently connected"""
        with self._lock:
            return self.client and self.client.is_connected if not self.simulation_mode else self._is_monitoring


# ============================================================================
# Global manager instance
# ============================================================================

_watch_manager: Optional[WatchIntegration] = None


def init_watch_integration(
    device_address: str = "C2:FC:28:B7:1C:1B",
    on_data_callback: Optional[Callable[[RoverVitals], None]] = None,
    simulation_mode: bool = False,
) -> WatchIntegration:
    """
    Initialize the global watch integration manager
    
    Args:
        device_address: BLE address of the watch
        on_data_callback: Optional callback for data updates
        simulation_mode: Force simulation mode (default: auto-detect)
    
    Returns:
        The watch integration manager
    """
    global _watch_manager
    
    if _watch_manager is not None:
        print("⚠️  Watch manager already initialized")
        return _watch_manager
    
    _watch_manager = WatchIntegration(
        device_address=device_address,
        on_data_callback=on_data_callback,
        simulation_mode=simulation_mode,
    )
    return _watch_manager


def get_watch_manager() -> Optional[WatchIntegration]:
    """Get the global watch integration manager"""
    return _watch_manager


def get_current_vitals() -> RoverVitals:
    """Get current vitals from the watch manager"""
    if _watch_manager:
        return _watch_manager.get_latest_vitals()
    return RoverVitals(timestamp=datetime.now())


if __name__ == "__main__":
    # Test the integration
    manager = init_watch_integration(simulation_mode=True)
    manager.start()
    
    print("\n📊 Listening for 10 seconds...\n")
    for i in range(10):
        vitals = manager.get_latest_vitals()
        print(f"[{i+1}] {vitals}")
        asyncio.sleep(1)
    
    manager.stop()
    print("\n✅ Test complete")
