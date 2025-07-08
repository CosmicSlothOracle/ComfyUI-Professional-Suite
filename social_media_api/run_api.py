#!/usr/bin/env python3
"""
Run script for Social Media Video Generation API
Starts the API server and ComfyUI dashboard
"""

from comfy_integration.comfy_ui_dashboard import start_dashboard
import os
import sys
import logging
import asyncio
import uvicorn
import argparse
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import dashboard


def setup_logging():
    """Set up logging configuration"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(
                f"{log_dir}/api_{datetime.now().strftime('%Y%m%d')}.log"),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


async def start_api_and_dashboard(host, port):
    """Start API server and ComfyUI dashboard"""
    logger = setup_logging()
    logger.info(f"Starting Social Media Video Generation API on {host}:{port}")

    # Start dashboard in background
    logger.info("Starting ComfyUI dashboard")
    dashboard = await start_dashboard()

    # Start API server
    config = uvicorn.Config(
        "api_server.social_media_api:app",
        host=host,
        port=port,
        log_level="info",
        reload=True
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Social Media Video Generation API")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Host to bind the API server to")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to bind the API server to")

    args = parser.parse_args()

    try:
        asyncio.run(start_api_and_dashboard(args.host, args.port))
    except KeyboardInterrupt:
        print("\nShutting down API server and dashboard")
    except Exception as e:
        print(f"Error starting API: {e}")
        sys.exit(1)
