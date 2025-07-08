"""
ComfyUI Dashboard Integration
Integrates the social media video generation API with ComfyUI
"""

from social_media_api.api_server.services.post_service import PostGenerationService
from social_media_api.api_server.services.video_service import VideoGenerationService
from social_media_api.api_server.services.nlp_service import NLPService
from social_media_api.api_server.services.trend_service import TrendAnalysisService
import os
import sys
import json
import logging
import asyncio
import websockets
import aiohttp
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
import uuid

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import API services


class ComfyUIDashboard:
    """ComfyUI Dashboard for Social Media Video Generation"""

    def __init__(self):
        """Initialize the ComfyUI dashboard"""
        self.logger = logging.getLogger(__name__)
        self.config_dir = "social_media_api/config"
        self.output_dir = "social_media_api/output"

        # ComfyUI connection settings
        self.comfy_server_address = "127.0.0.1"
        self.comfy_server_port = 8188
        self.comfy_ws_url = f"ws://{self.comfy_server_address}:{self.comfy_server_port}/ws"
        self.comfy_api_url = f"http://{self.comfy_server_address}:{self.comfy_server_port}/api"

        # Services
        self.trend_service = None
        self.nlp_service = None
        self.video_service = None
        self.post_service = None

        # Status tracking
        self.status = {
            "connected": False,
            "last_update": None,
            "scheduled_task_active": False,
            "last_scheduled_run": None,
            "error": None
        }

        # Task tracking
        self.tasks = {}

        # Callbacks
        self.status_callbacks = []
        self.task_callbacks = []

        # Load configuration
        self.config = self._load_config()

        # Initialize services
        self._initialize_services()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        config_path = os.path.join(self.config_dir, "dashboard_config.json")
        default_config = {
            "comfy_server_address": "127.0.0.1",
            "comfy_server_port": 8188,
            "auto_connect": True,
            "scheduled_task_time": "21:30",
            "platforms": ["tiktok", "instagram"],
            "region": "DE",
            "age_range": [16, 27],
            "video_style": "modern",
            "video_resolution": "1080x1920",
            "video_duration": 30,
            "include_music": True,
            "include_voiceover": True,
            "language": "de",
            "post_style": "viral",
            "hashtag_count": 5,
            "include_emojis": True
        }

        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    # Update with any missing default values
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                    return config
            else:
                # Create default config
                os.makedirs(self.config_dir, exist_ok=True)
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(default_config, f, ensure_ascii=False, indent=2)
                return default_config
        except Exception as e:
            self.logger.error(f"Error loading dashboard config: {e}")
            return default_config

    def _initialize_services(self):
        """Initialize API services"""
        try:
            self.trend_service = TrendAnalysisService()
            self.nlp_service = NLPService()
            self.video_service = VideoGenerationService()
            self.post_service = PostGenerationService()
            self.logger.info("API services initialized")
        except Exception as e:
            self.logger.error(f"Error initializing API services: {e}")
            self.status["error"] = f"Service initialization error: {str(e)}"

    async def connect_to_comfy_ui(self):
        """Connect to ComfyUI WebSocket server"""
        try:
            self.logger.info(f"Connecting to ComfyUI at {self.comfy_ws_url}")

            async with websockets.connect(self.comfy_ws_url) as websocket:
                self.status["connected"] = True
                self.status["last_update"] = datetime.now().isoformat()
                self.status["error"] = None
                self._notify_status_update()

                self.logger.info("Connected to ComfyUI")

                # Main connection loop
                while True:
                    try:
                        message = await websocket.recv()
                        await self._handle_comfy_message(message)
                    except websockets.exceptions.ConnectionClosed:
                        self.logger.warning("ComfyUI connection closed")
                        self.status["connected"] = False
                        self.status["error"] = "Connection closed"
                        self._notify_status_update()
                        break
                    except Exception as e:
                        self.logger.error(
                            f"Error handling ComfyUI message: {e}")
                        self.status["error"] = f"Message handling error: {str(e)}"
                        self._notify_status_update()

        except Exception as e:
            self.logger.error(f"Error connecting to ComfyUI: {e}")
            self.status["connected"] = False
            self.status["error"] = f"Connection error: {str(e)}"
            self._notify_status_update()

            # Try to reconnect after delay
            await asyncio.sleep(5)
            asyncio.create_task(self.connect_to_comfy_ui())

    async def _handle_comfy_message(self, message: str):
        """Handle message from ComfyUI WebSocket"""
        try:
            data = json.loads(message)

            # Update status
            self.status["last_update"] = datetime.now().isoformat()
            self._notify_status_update()

            # Handle different message types
            if "type" in data:
                if data["type"] == "status":
                    # Status update
                    pass
                elif data["type"] == "execution_start":
                    # Workflow execution started
                    pass
                elif data["type"] == "execution_complete":
                    # Workflow execution completed
                    pass
                elif data["type"] == "error":
                    # Error message
                    self.status["error"] = data.get("message", "Unknown error")
                    self._notify_status_update()

        except Exception as e:
            self.logger.error(f"Error parsing ComfyUI message: {e}")

    async def send_comfy_workflow(self, workflow: Dict[str, Any]) -> bool:
        """
        Send workflow to ComfyUI for execution

        Args:
            workflow: Workflow definition

        Returns:
            Success status
        """
        try:
            async with aiohttp.ClientSession() as session:
                # Queue the workflow
                async with session.post(
                    f"{self.comfy_api_url}/prompt",
                    json={"prompt": workflow}
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        self.logger.info(f"Workflow queued: {result}")
                        return True
                    else:
                        error_text = await response.text()
                        self.logger.error(
                            f"Error queuing workflow: {error_text}")
                        return False

        except Exception as e:
            self.logger.error(f"Error sending workflow to ComfyUI: {e}")
            return False

    async def run_video_generation_workflow(
        self,
        trend_data: Dict[str, Any],
        workflow_type: str = "default"
    ) -> Optional[str]:
        """
        Run video generation workflow in ComfyUI

        Args:
            trend_data: Trend data
            workflow_type: Type of workflow to run

        Returns:
            Task ID if successful, None otherwise
        """
        try:
            # Create task
            task_id = str(uuid.uuid4())

            # Initialize task status
            self.tasks[task_id] = {
                "task_id": task_id,
                "type": "video_generation",
                "status": "initializing",
                "progress": 0.0,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "trend_data": trend_data,
                "workflow_type": workflow_type,
                "result": None,
                "error": None
            }

            self._notify_task_update(task_id)

            # Start task in background
            asyncio.create_task(self._process_video_generation(
                task_id, trend_data, workflow_type))

            return task_id

        except Exception as e:
            self.logger.error(f"Error starting video generation workflow: {e}")
            return None

    async def _process_video_generation(
        self,
        task_id: str,
        trend_data: Dict[str, Any],
        workflow_type: str
    ):
        """Process video generation in background"""
        try:
            # Update task status
            self.tasks[task_id]["status"] = "processing"
            self.tasks[task_id]["progress"] = 0.1
            self.tasks[task_id]["updated_at"] = datetime.now().isoformat()
            self._notify_task_update(task_id)

            # Perform NLP analysis
            self.logger.info(f"Performing NLP analysis for task {task_id}")
            nlp_analysis = await self.nlp_service.analyze_trend(trend_data)

            if not nlp_analysis:
                self.logger.error(f"NLP analysis failed for task {task_id}")
                self.tasks[task_id]["status"] = "failed"
                self.tasks[task_id]["error"] = "NLP analysis failed"
                self.tasks[task_id]["updated_at"] = datetime.now().isoformat()
                self._notify_task_update(task_id)
                return

            # Update progress
            self.tasks[task_id]["progress"] = 0.3
            self.tasks[task_id]["updated_at"] = datetime.now().isoformat()
            self._notify_task_update(task_id)

            # Generate video
            self.logger.info(f"Generating video for task {task_id}")
            video_result = await self.video_service.generate_video(
                trend_data=trend_data,
                nlp_analysis=nlp_analysis,
                style=self.config["video_style"],
                resolution=self.config["video_resolution"],
                duration=self.config["video_duration"],
                include_music=self.config["include_music"],
                include_voiceover=self.config["include_voiceover"],
                language=self.config["language"]
            )

            if not video_result:
                self.logger.error(
                    f"Video generation failed for task {task_id}")
                self.tasks[task_id]["status"] = "failed"
                self.tasks[task_id]["error"] = "Video generation failed"
                self.tasks[task_id]["updated_at"] = datetime.now().isoformat()
                self._notify_task_update(task_id)
                return

            # Update progress
            self.tasks[task_id]["progress"] = 0.7
            self.tasks[task_id]["updated_at"] = datetime.now().isoformat()
            self._notify_task_update(task_id)

            # Generate post text
            self.logger.info(f"Generating post text for task {task_id}")
            video_id = video_result.get("video_id", str(uuid.uuid4()))
            post_result = await self.post_service.generate_post_text(
                trend_data=trend_data,
                video_id=video_id,
                style=self.config["post_style"],
                hashtag_count=self.config["hashtag_count"],
                include_emojis=self.config["include_emojis"]
            )

            if not post_result:
                self.logger.error(
                    f"Post text generation failed for task {task_id}")
                self.tasks[task_id]["status"] = "failed"
                self.tasks[task_id]["error"] = "Post text generation failed"
                self.tasks[task_id]["updated_at"] = datetime.now().isoformat()
                self._notify_task_update(task_id)
                return

            # Update task with results
            self.tasks[task_id]["status"] = "completed"
            self.tasks[task_id]["progress"] = 1.0
            self.tasks[task_id]["result"] = {
                "trend_data": trend_data,
                "nlp_analysis": nlp_analysis,
                "video": video_result,
                "post": post_result
            }
            self.tasks[task_id]["updated_at"] = datetime.now().isoformat()
            self._notify_task_update(task_id)

            self.logger.info(
                f"Video generation workflow completed for task {task_id}")

        except Exception as e:
            self.logger.error(f"Error in video generation workflow: {e}")
            self.tasks[task_id]["status"] = "failed"
            self.tasks[task_id]["error"] = str(e)
            self.tasks[task_id]["updated_at"] = datetime.now().isoformat()
            self._notify_task_update(task_id)

    async def run_scheduled_task(self):
        """Run the scheduled daily task"""
        try:
            self.status["scheduled_task_active"] = True
            self.status["last_scheduled_run"] = datetime.now().isoformat()
            self._notify_status_update()

            self.logger.info("Running scheduled daily task")

            # Analyze trends
            trends = await self.trend_service.analyze_trends(
                platforms=self.config["platforms"],
                region=self.config["region"],
                age_range=self.config["age_range"],
                limit=5,
                include_hashtags=True,
                include_sounds=True,
                include_formats=True
            )

            if not trends or len(trends) == 0:
                self.logger.error("No trends found in scheduled task")
                self.status["scheduled_task_active"] = False
                self.status["error"] = "No trends found"
                self._notify_status_update()
                return

            # Select top trend
            top_trend = trends[0]

            # Run video generation workflow
            task_id = await self.run_video_generation_workflow(top_trend)

            if not task_id:
                self.logger.error(
                    "Failed to start video generation workflow in scheduled task")
                self.status["scheduled_task_active"] = False
                self.status["error"] = "Failed to start workflow"
                self._notify_status_update()
                return

            # Wait for task to complete
            while task_id in self.tasks and self.tasks[task_id]["status"] not in ["completed", "failed"]:
                await asyncio.sleep(1)

            # Check result
            if task_id in self.tasks and self.tasks[task_id]["status"] == "completed":
                self.logger.info("Scheduled task completed successfully")

                # Save report
                result = self.tasks[task_id]["result"]
                report_path = os.path.join(
                    self.output_dir, "reports", f"daily_report_{datetime.now().strftime('%Y%m%d')}.json")
                os.makedirs(os.path.dirname(report_path), exist_ok=True)

                with open(report_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
            else:
                self.logger.error("Scheduled task failed")
                self.status["error"] = "Scheduled task failed"

            self.status["scheduled_task_active"] = False
            self._notify_status_update()

        except Exception as e:
            self.logger.error(f"Error in scheduled task: {e}")
            self.status["scheduled_task_active"] = False
            self.status["error"] = f"Scheduled task error: {str(e)}"
            self._notify_status_update()

    def register_status_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Register callback for status updates"""
        self.status_callbacks.append(callback)

    def register_task_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Register callback for task updates"""
        self.task_callbacks.append(callback)

    def _notify_status_update(self):
        """Notify all registered status callbacks"""
        for callback in self.status_callbacks:
            try:
                callback(self.status)
            except Exception as e:
                self.logger.error(f"Error in status callback: {e}")

    def _notify_task_update(self, task_id: str):
        """Notify all registered task callbacks"""
        if task_id in self.tasks:
            for callback in self.task_callbacks:
                try:
                    callback(task_id, self.tasks[task_id])
                except Exception as e:
                    self.logger.error(f"Error in task callback: {e}")

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task by ID"""
        return self.tasks.get(task_id)

    def get_all_tasks(self) -> Dict[str, Dict[str, Any]]:
        """Get all tasks"""
        return self.tasks

    def get_status(self) -> Dict[str, Any]:
        """Get current status"""
        return self.status

    def update_config(self, new_config: Dict[str, Any]) -> bool:
        """Update dashboard configuration"""
        try:
            # Update config
            for key, value in new_config.items():
                if key in self.config:
                    self.config[key] = value

            # Save to file
            config_path = os.path.join(
                self.config_dir, "dashboard_config.json")
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)

            return True
        except Exception as e:
            self.logger.error(f"Error updating config: {e}")
            return False


async def start_dashboard():
    """Start the ComfyUI dashboard"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("social_media_api/logs/dashboard.log"),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting ComfyUI dashboard")

    # Create dashboard
    dashboard = ComfyUIDashboard()

    # Example status callback
    def status_updated(status):
        logger.info(f"Dashboard status updated: {status['connected']}")

    # Example task callback
    def task_updated(task_id, task):
        logger.info(f"Task {task_id} updated: {task['status']}")

    # Register callbacks
    dashboard.register_status_callback(status_updated)
    dashboard.register_task_callback(task_updated)

    # Connect to ComfyUI
    asyncio.create_task(dashboard.connect_to_comfy_ui())

    # Set up scheduled task
    async def schedule_daily_task():
        while True:
            now = datetime.now()
            scheduled_time = dashboard.config["scheduled_task_time"].split(":")
            target_hour = int(scheduled_time[0])
            target_minute = int(scheduled_time[1])

            # Calculate time until next run
            target_time = now.replace(
                hour=target_hour, minute=target_minute, second=0, microsecond=0)
            if now >= target_time:
                target_time = target_time.replace(day=target_time.day + 1)

            # Wait until scheduled time
            wait_seconds = (target_time - now).total_seconds()
            logger.info(f"Scheduled task will run in {wait_seconds} seconds")
            await asyncio.sleep(wait_seconds)

            # Run the task
            await dashboard.run_scheduled_task()

    # Start the scheduler
    asyncio.create_task(schedule_daily_task())

    return dashboard


if __name__ == "__main__":
    asyncio.run(start_dashboard())
