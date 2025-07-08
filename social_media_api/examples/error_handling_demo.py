#!/usr/bin/env python3
"""
Error Handling Demonstration for Social Media Video Generation API
This script demonstrates how the system handles various error scenarios.
"""

from api_server.services.post_service import PostGenerationService
from api_server.services.video_service import VideoGenerationService
from api_server.services.nlp_service import NLPService
from api_server.services.trend_service import TrendAnalysisService
import os
import sys
import asyncio
import json
import logging
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import services


class ErrorHandlingDemo:
    """Demonstrates error handling capabilities"""

    def __init__(self):
        """Initialize the demo"""
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Initialize services
        self.trend_service = TrendAnalysisService()
        self.nlp_service = NLPService()
        self.video_service = VideoGenerationService()
        self.post_service = PostGenerationService()

        # Create output directory
        os.makedirs("examples/error_logs", exist_ok=True)

    async def demo_empty_trend_data(self):
        """Demonstrate handling of empty trend data"""
        self.logger.info("=== DEMO: Empty Trend Data ===")

        try:
            # Simulate empty trend data
            empty_trends = []

            self.logger.info("Analyzing empty trend data...")

            if not empty_trends:
                error_message = "No trends found. Cannot proceed with video generation."
                self.logger.error(error_message)

                # Log the error
                self._log_error("empty_trend_data", error_message)

                # In a real system, this would trigger a notification
                self.logger.info("Notification would be sent to administrator")

                return False

        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            self._log_error("empty_trend_data", str(e))
            return False

        return True

    async def demo_invalid_trend_data(self):
        """Demonstrate handling of invalid trend data"""
        self.logger.info("=== DEMO: Invalid Trend Data ===")

        try:
            # Simulate invalid trend data (missing required fields)
            invalid_trend = {
                "platform": "tiktok",
                # Missing id, name, and other required fields
            }

            self.logger.info(
                "Performing NLP analysis on invalid trend data...")

            # Validate trend data before processing
            if "id" not in invalid_trend or "name" not in invalid_trend:
                error_message = "Invalid trend data: Missing required fields (id, name)"
                self.logger.error(error_message)

                # Log the error
                self._log_error("invalid_trend_data", error_message)

                # In a real system, this would trigger a notification
                self.logger.info("Notification would be sent to administrator")

                return False

            # This code should not be reached due to validation
            nlp_analysis = await self.nlp_service.analyze_trend(invalid_trend)

        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            self._log_error("invalid_trend_data", str(e))
            return False

        return True

    async def demo_api_failure(self):
        """Demonstrate handling of API failure"""
        self.logger.info("=== DEMO: API Failure ===")

        try:
            # Get a valid trend for testing
            trends = await self.trend_service.get_mock_trends_for_testing()
            if not trends:
                self.logger.error("Could not get test trends")
                return False

            trend = trends[0]

            # Perform NLP analysis
            nlp_analysis = await self.nlp_service.analyze_trend(trend)
            if not nlp_analysis:
                self.logger.error("NLP analysis failed")
                return False

            # Simulate API failure in video generation
            self.logger.info("Simulating VEO3 API failure...")

            # Modify credentials to force API failure
            original_credentials = self.video_service.credentials
            self.video_service.credentials = None

            try:
                # This should fail due to missing credentials
                video_result = await self.video_service.generate_video(
                    trend_data=trend,
                    nlp_analysis=nlp_analysis,
                    style="modern",
                    resolution="1080x1920",
                    duration=30
                )

                if not video_result:
                    error_message = "Video generation failed due to API error"
                    self.logger.error(error_message)

                    # Log the error
                    self._log_error("api_failure", error_message)

                    # In a real system, this would trigger a notification
                    self.logger.info(
                        "Notification would be sent to administrator")
            finally:
                # Restore original credentials
                self.video_service.credentials = original_credentials

        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            self._log_error("api_failure", str(e))
            return False

        return True

    async def demo_partial_failure(self):
        """Demonstrate handling of partial failure (post generation fails)"""
        self.logger.info("=== DEMO: Partial Failure ===")

        try:
            # Get a valid trend for testing
            trends = await self.trend_service.get_mock_trends_for_testing()
            if not trends:
                self.logger.error("Could not get test trends")
                return False

            trend = trends[0]

            # Perform NLP analysis
            nlp_analysis = await self.nlp_service.analyze_trend(trend)
            if not nlp_analysis:
                self.logger.error("NLP analysis failed")
                return False

            # Generate video
            video_result = await self.video_service.generate_video(
                trend_data=trend,
                nlp_analysis=nlp_analysis,
                style="modern",
                resolution="1080x1920",
                duration=30
            )

            if not video_result:
                self.logger.error("Video generation failed")
                return False

            # Simulate failure in post generation
            self.logger.info("Simulating post generation failure...")

            # Pass invalid video_id to cause failure
            try:
                post_result = await self.post_service.generate_post_text(
                    trend_data={},  # Invalid trend data
                    video_id=video_result.get("video_id", "")
                )

                if not post_result:
                    error_message = "Post generation failed but video was created"
                    self.logger.error(error_message)

                    # Log the error
                    self._log_error("partial_failure", error_message)

                    # In a real system, this would trigger a notification
                    self.logger.info(
                        "Notification would be sent to administrator")
                    self.logger.info(
                        "Video result would be preserved for manual post creation")

                    # Save the video metadata for recovery
                    with open("examples/error_logs/recovered_video.json", "w", encoding="utf-8") as f:
                        json.dump(video_result, f, indent=2,
                                  ensure_ascii=False)
            except Exception as e:
                self.logger.error(f"Post generation error: {e}")
                self._log_error("partial_failure",
                                f"Post generation error: {e}")

        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            self._log_error("partial_failure", str(e))
            return False

        return True

    def _log_error(self, error_type, error_message):
        """Log error to file"""
        error_log = {
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "error_message": error_message
        }

        log_file = f"examples/error_logs/{error_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(error_log, f, indent=2, ensure_ascii=False)


async def run_error_handling_demo():
    """Run all error handling demonstrations"""
    demo = ErrorHandlingDemo()

    # Run all demos
    await demo.demo_empty_trend_data()
    await demo.demo_invalid_trend_data()
    await demo.demo_api_failure()
    await demo.demo_partial_failure()

    print("\nError handling demonstration completed. Check the logs for details.")


if __name__ == "__main__":
    asyncio.run(run_error_handling_demo())
