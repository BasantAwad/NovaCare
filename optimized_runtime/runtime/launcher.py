"""
Main Runtime Launcher

Entry point for the optimized NovaCare runtime.
Initializes all subsystems and starts the orchestrator.

Usage:
    python -m optimized_runtime.runtime.launcher [--mode serbot|laptop|debug]
"""

import asyncio
import logging
import argparse
from typing import Optional

from ..orchestrator import RuntimeOrchestrator
from ..adapters import get_service_registry
from ..state import get_robot_state

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RuntimeLauncher:
    """Manages runtime lifecycle"""
    
    def __init__(self, mode: str = "debug"):
        self.mode = mode  # serbot, laptop, debug
        self.orchestrator: Optional[RuntimeOrchestrator] = None
        self.registry = get_service_registry()
    
    async def startup(self):
        """Initialize and start all systems"""
        logger.info(f"Starting NovaCare optimized runtime (mode={self.mode})")
        
        try:
            # Initialize service registry
            await self.registry.initialize_all()
            
            # Check service health
            health = await self.registry.health_check_all()
            logger.info(f"Service health: {health}")
            
            # Initialize orchestrator
            self.orchestrator = RuntimeOrchestrator()
            await self.orchestrator.initialize()
            
            # Start orchestrator
            logger.info("Starting runtime orchestrator...")
            await self.orchestrator.start()
        
        except Exception as e:
            logger.error(f"Startup failed: {e}")
            raise
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down runtime...")
        
        try:
            if self.orchestrator:
                await self.orchestrator.stop()
            
            await self.registry.shutdown_all()
            logger.info("Shutdown complete")
        
        except Exception as e:
            logger.error(f"Shutdown error: {e}")
    
    async def run(self):
        """Main runtime loop"""
        try:
            await self.startup()
            
            # Keep running
            while True:
                await asyncio.sleep(1)
                
                # Optional: periodic health checks
                if self.mode == "debug":
                    metrics = self.orchestrator.get_metrics()
                    logger.debug(f"Metrics: {metrics}")
        
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        
        finally:
            await self.shutdown()


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="NovaCare Optimized Runtime Launcher"
    )
    parser.add_argument(
        "--mode",
        choices=["serbot", "laptop", "debug"],
        default="debug",
        help="Runtime mode"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Adjust logging
    logging.getLogger().setLevel(args.log_level)
    
    launcher = RuntimeLauncher(mode=args.mode)
    await launcher.run()


if __name__ == "__main__":
    asyncio.run(main())
