#!/usr/bin/env python3
"""
Example Workflow for Social Media Video Generation API
This script demonstrates a complete workflow from trend analysis to video generation.
"""

from api_server.services.post_service import PostGenerationService
from api_server.services.video_service import VideoGenerationService
from api_server.services.nlp_service import NLPService
from api_server.services.trend_service import TrendAnalysisService
import os
import sys
import asyncio
import json
from datetime import datetime
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import services


async def run_example_workflow():
    """Run a complete example workflow"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    logger.info("Starting example workflow")

    # Initialize services
    trend_service = TrendAnalysisService()
    nlp_service = NLPService()
    video_service = VideoGenerationService()
    post_service = PostGenerationService()

    # Step 1: Analyze trends
    logger.info("Step 1: Analyzing trends")
    try:
        # For demonstration purposes, we'll use mock trends
        trends = await trend_service.get_mock_trends_for_testing()

        if not trends:
            logger.error("No trends found. Stopping workflow.")
            return

        logger.info(f"Found {len(trends)} trends")
        top_trend = trends[0]  # Select the top trend
        logger.info(f"Selected top trend: {top_trend.get('name', 'Unknown')}")

        # Save trend data
        os.makedirs("examples/output", exist_ok=True)
        with open("examples/output/trend_data.json", "w", encoding="utf-8") as f:
            json.dump(top_trend, f, indent=2, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Error in trend analysis: {e}")
        return

    # Step 2: NLP Analysis
    logger.info("Step 2: Performing NLP analysis")
    try:
        nlp_analysis = await nlp_service.analyze_trend(top_trend)

        if not nlp_analysis:
            logger.error("NLP analysis failed. Stopping workflow.")
            return

        logger.info(
            f"NLP analysis completed: {nlp_analysis.get('classification', {}).get('primary_type', 'Unknown')} content")

        # Save NLP analysis
        with open("examples/output/nlp_analysis.json", "w", encoding="utf-8") as f:
            json.dump(nlp_analysis, f, indent=2, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Error in NLP analysis: {e}")
        return

    # Step 3: Generate video
    logger.info("Step 3: Generating video")
    try:
        video_result = await video_service.generate_video(
            trend_data=top_trend,
            nlp_analysis=nlp_analysis,
            style="modern",
            resolution="1080x1920",
            duration=30,
            include_music=True,
            include_voiceover=True,
            language="de"
        )

        if not video_result:
            logger.error("Video generation failed. Stopping workflow.")
            return

        logger.info(
            f"Video generated: {video_result.get('video_path', 'Unknown')}")

        # Save video metadata
        with open("examples/output/video_metadata.json", "w", encoding="utf-8") as f:
            # Remove large script data to keep the output clean
            if "script" in video_result:
                video_result["script"] = "Script content removed for brevity"
            json.dump(video_result, f, indent=2, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Error in video generation: {e}")
        return

    # Step 4: Generate post text
    logger.info("Step 4: Generating post text")
    try:
        video_id = video_result.get("video_id", "unknown")
        post_result = await post_service.generate_post_text(
            trend_data=top_trend,
            video_id=video_id,
            style="viral",
            hashtag_count=5,
            include_emojis=True
        )

        if not post_result:
            logger.error("Post text generation failed.")
            return

        logger.info(
            f"Post text generated: {post_result.get('main_text', 'Unknown')}")

        # Save post data
        with open("examples/output/post_data.json", "w", encoding="utf-8") as f:
            json.dump(post_result, f, indent=2, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Error in post text generation: {e}")
        return

    # Step 5: Create final report
    logger.info("Step 5: Creating final report")
    try:
        report = {
            "timestamp": datetime.now().isoformat(),
            "trend": top_trend,
            "nlp_analysis": {
                "sentiment": nlp_analysis.get("sentiment", {}),
                "classification": nlp_analysis.get("classification", {}),
                "content_mechanics": nlp_analysis.get("content_mechanics", [])
            },
            "video": {
                "video_id": video_result.get("video_id", ""),
                "duration": video_result.get("duration", 0),
                "resolution": video_result.get("resolution", ""),
                "style": video_result.get("style", "")
            },
            "post": {
                "main_text": post_result.get("main_text", ""),
                "hashtags": post_result.get("hashtags", []),
                "complete_text": post_result.get("complete_text", "")
            }
        }

        with open("examples/output/final_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info("Final report created")

    except Exception as e:
        logger.error(f"Error creating final report: {e}")
        return

    logger.info("Example workflow completed successfully")


if __name__ == "__main__":
    asyncio.run(run_example_workflow())
