"""
TEST APPLICATION - Simple example of integrating HRYFINE watch data
"""

import asyncio
import json
from pathlib import Path
from watch_client import HRYFINEWatchClient
from watch_protocol import WatchData


class WatchDataCollector:
    """Collect and display watch sensor data"""
    
    def __init__(self, output_file: str = "watch_data.json"):
        self.output_file = output_file
        self.data_log = []
        self.current_stats = {
            'total_steps': 0,
            'avg_heart_rate': 0,
            'min_heart_rate': 999,
            'max_heart_rate': 0,
            'heart_rate_readings': 0,
        }
    
    def on_data(self, watch_data: WatchData):
        """Callback for incoming watch data"""
        
        # Log raw data
        log_entry = {
            'timestamp': watch_data.timestamp.isoformat(),
            'heart_rate': watch_data.heart_rate,
            'steps': watch_data.steps,
            'battery': watch_data.battery,
        }
        self.data_log.append(log_entry)
        
        # Update statistics
        if watch_data.heart_rate:
            self.current_stats['avg_heart_rate'] = (
                (self.current_stats['avg_heart_rate'] * self.current_stats['heart_rate_readings'] + watch_data.heart_rate) /
                (self.current_stats['heart_rate_readings'] + 1)
            )
            self.current_stats['min_heart_rate'] = min(
                self.current_stats['min_heart_rate'],
                watch_data.heart_rate
            )
            self.current_stats['max_heart_rate'] = max(
                self.current_stats['max_heart_rate'],
                watch_data.heart_rate
            )
            self.current_stats['heart_rate_readings'] += 1
        
        if watch_data.steps:
            self.current_stats['total_steps'] = watch_data.steps
        
        # Pretty print
        print(f"\n{'='*60}")
        print(f"⏰ {watch_data.timestamp.strftime('%H:%M:%S.%f')[:-3]}")
        print(f"{'='*60}")
        
        if watch_data.heart_rate:
            hr_zone = self._get_hr_zone(watch_data.heart_rate)
            print(f"❤️  Heart Rate:  {watch_data.heart_rate} BPM  {hr_zone}")
        
        if watch_data.steps:
            print(f"👟 Steps:       {watch_data.steps}")
        
        if watch_data.battery is not None:
            emoji = self._get_battery_emoji(watch_data.battery)
            print(f"🔋 Battery:     {watch_data.battery}%  {emoji}")
        
        # Show live stats
        print(f"\n📊 SESSION STATS:")
        print(f"  Total Steps:    {self.current_stats['total_steps']}")
        print(f"  Avg HR:         {self.current_stats['avg_heart_rate']:.1f} BPM")
        print(f"  Min HR:         {self.current_stats['min_heart_rate']} BPM")
        print(f"  Max HR:         {self.current_stats['max_heart_rate']} BPM")
        print(f"  HR Readings:    {self.current_stats['heart_rate_readings']}")
    
    @staticmethod
    def _get_hr_zone(heart_rate: int) -> str:
        """Get heart rate zone emoji/name"""
        if heart_rate < 50:
            return "🟦 Zone 1 (Recovery)"
        elif heart_rate < 100:
            return "🟩 Zone 2 (Cardio)"
        elif heart_rate < 150:
            return "🟨 Zone 3 (Tempo)"
        elif heart_rate < 180:
            return "🟧 Zone 4 (Threshold)"
        else:
            return "🟥 Zone 5 (Max)"
    
    @staticmethod
    def _get_battery_emoji(battery: int) -> str:
        """Battery level emoji"""
        if battery >= 80:
            return "🔋"
        elif battery >= 50:
            return "🔋"
        elif battery >= 20:
            return "🪫"
        else:
            return "⚠️"
    
    def save_log(self):
        """Save data to JSON file"""
        with open(self.output_file, 'w') as f:
            json.dump(self.data_log, f, indent=2)
        print(f"\n💾 Saved {len(self.data_log)} records to {self.output_file}")


async def main():
    """Run the test application"""
    
    # CHANGE THIS TO YOUR WATCH ADDRESS
    WATCH_ADDRESS = "C2:FC:28:B7:1C:1B"
    
    print("=" * 60)
    print("HRYFINE WATCH - TEST APPLICATION")
    print("=" * 60)
    print(f"\nConnecting to {WATCH_ADDRESS}...")
    
    # Create client and collector
    client = HRYFINEWatchClient(WATCH_ADDRESS)
    collector = WatchDataCollector("watch_session.json")
    
    # Connect
    if not await client.connect():
        print("Failed to connect!")
        return
    
    print("\n✅ Connected! Monitoring watch for 60 seconds...")
    print("(Move around, check your heart rate, etc.)\n")
    
    # Start monitoring with a timeout
    try:
        # Create monitoring task with timeout
        monitor_task = asyncio.create_task(
            client.start_monitoring(collector.on_data)
        )
        
        # Wait for 60 seconds or until task completes
        await asyncio.wait_for(monitor_task, timeout=60)
    
    except asyncio.TimeoutError:
        print("\n⏰ 60 seconds elapsed, stopping...")
    except KeyboardInterrupt:
        print("\n⏹️ Stopped by user")
    finally:
        await client.disconnect()
        collector.save_log()
        
        # Print final summary
        print("\n" + "=" * 60)
        print("FINAL SUMMARY")
        print("=" * 60)
        print(f"Total Steps:        {collector.current_stats['total_steps']}")
        print(f"Avg Heart Rate:     {collector.current_stats['avg_heart_rate']:.1f} BPM")
        print(f"Min Heart Rate:     {collector.current_stats['min_heart_rate']} BPM")
        print(f"Max Heart Rate:     {collector.current_stats['max_heart_rate']} BPM")
        print(f"HR Readings:        {collector.current_stats['heart_rate_readings']}")
        print(f"Total Records:      {len(collector.data_log)}")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
