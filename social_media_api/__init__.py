"""
Social Media Video Generation API
A professional, documented, robust API pipeline with ComfyUI and Google VEO3
for daily, AI-based generation of social media videos based on current trends.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

import os
import logging

# Ensure directories exist
os.makedirs("social_media_api/logs", exist_ok=True)
os.makedirs("social_media_api/output/videos", exist_ok=True)
os.makedirs("social_media_api/output/reports", exist_ok=True)
os.makedirs("social_media_api/temp", exist_ok=True)
os.makedirs("social_media_api/config", exist_ok=True)

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
