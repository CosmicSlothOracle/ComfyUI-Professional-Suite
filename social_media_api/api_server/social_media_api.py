#!/usr/bin/env python3
"""
🚀 Social Media Video Generation API Server
FastAPI-basierte REST-Schnittstelle für automatisierte Social-Media-Video-Generierung
"""

import os
import asyncio
import logging
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import tempfile
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

# Import our services
from social_media_api.api_server.services.trend_service import TrendAnalysisService
from social_media_api.api_server.services.nlp_service import NLPService
from social_media_api.api_server.services.video_service import VideoGenerationService
from social_media_api.api_server.services.post_service import PostGenerationService

# API Models


class TrendAnalysisRequest(BaseModel):
    """Request model for trend analysis"""
    platforms: List[str] = ["tiktok", "instagram"]
    region: str = "DE"
    age_range: List[int] = [16, 27]
    limit: int = 10
    include_hashtags: bool = True
    include_sounds: bool = True
    include_formats: bool = True


class VideoGenerationRequest(BaseModel):
    """Request model for video generation"""
    trend_id: str
    style: str = "modern"
    resolution: str = "1080x1920"
    duration: int = Field(default=30, ge=10, le=60)
    include_music: bool = True
    include_voiceover: bool = True
    language: str = "de"


class PostTextRequest(BaseModel):
    """Request model for post text generation"""
    video_id: str
    trend_id: str
    style: str = "viral"
    hashtag_count: int = Field(default=5, ge=1, le=10)
    include_emojis: bool = True


class TaskStatus(BaseModel):
    """Task status model"""
    task_id: str
    status: str  # pending, processing, completed, failed
    progress: float  # 0.0 - 1.0
    created_at: datetime
    updated_at: datetime
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


# Global state management
task_store = {}
app = FastAPI(
    title="Social Media Video Generator API",
    description="🎬 API für die automatisierte Generierung von Social-Media-Videos basierend auf aktuellen Trends",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global service instances
trend_service = None
nlp_service = None
video_service = None
post_service = None


def get_trend_service():
    """Trend Service Dependency"""
    global trend_service
    if trend_service is None:
        trend_service = TrendAnalysisService()
    return trend_service


def get_nlp_service():
    """NLP Service Dependency"""
    global nlp_service
    if nlp_service is None:
        nlp_service = NLPService()
    return nlp_service


def get_video_service():
    """Video Service Dependency"""
    global video_service
    if video_service is None:
        video_service = VideoGenerationService()
    return video_service


def get_post_service():
    """Post Service Dependency"""
    global post_service
    if post_service is None:
        post_service = PostGenerationService()
    return post_service


@app.on_event("startup")
async def startup_event():
    """Server initialization"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("social_media_api/logs/api.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("Social Media Video Generator API starting up...")

    # Create output directories
    os.makedirs("social_media_api/output/videos", exist_ok=True)
    os.makedirs("social_media_api/output/reports", exist_ok=True)
    os.makedirs("social_media_api/logs", exist_ok=True)
    os.makedirs("social_media_api/temp", exist_ok=True)

    # Initialize services
    get_trend_service()
    get_nlp_service()
    get_video_service()
    get_post_service()

    logger.info("API ready!")


@app.get("/", tags=["General"])
async def root():
    """API status and information"""
    return {
        "message": "🎬 Social Media Video Generator API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "analyze_trends": "/trends/analyze",
            "generate_video": "/video/generate",
            "generate_post": "/post/generate",
            "task_status": "/tasks/{task_id}",
            "download": "/download/{filename}"
        },
        "docs": "/docs"
    }


@app.get("/health", tags=["General"])
async def health_check():
    """Health Check Endpoint"""
    try:
        # Check GPU status if applicable
        gpu_available = False
        gpu_memory = 0

        try:
            import torch
            gpu_available = torch.cuda.is_available()
            if gpu_available:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
        except:
            pass

        # Check service status
        trend_service_status = "available" if trend_service is not None else "unavailable"
        nlp_service_status = "available" if nlp_service is not None else "unavailable"
        video_service_status = "available" if video_service is not None else "unavailable"
        post_service_status = "available" if post_service is not None else "unavailable"

        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "gpu_available": gpu_available,
            "gpu_memory_gb": gpu_memory / (1024**3) if gpu_memory > 0 else 0,
            "active_tasks": len([t for t in task_store.values() if t.status == "processing"]),
            "services": {
                "trend_analysis": trend_service_status,
                "nlp_analysis": nlp_service_status,
                "video_generation": video_service_status,
                "post_generation": post_service_status
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Health check failed: {str(e)}")


@app.post("/trends/analyze", tags=["Trends"])
async def analyze_trends(
    background_tasks: BackgroundTasks,
    request: TrendAnalysisRequest,
    trend_service: TrendAnalysisService = Depends(get_trend_service)
):
    """Analyze current social media trends"""

    # Generate task ID
    task_id = str(uuid.uuid4())

    # Initialize task status
    task_store[task_id] = TaskStatus(
        task_id=task_id,
        status="processing",
        progress=0.0,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )

    # Start background task
    background_tasks.add_task(
        process_trend_analysis,
        task_id,
        request,
        trend_service
    )

    return {
        "task_id": task_id,
        "status": "processing",
        "message": "Trend analysis started"
    }


async def process_trend_analysis(
    task_id: str,
    request: TrendAnalysisRequest,
    trend_service: TrendAnalysisService
):
    """Process trend analysis in background"""
    try:
        # Update task status
        task_store[task_id].progress = 0.1
        task_store[task_id].updated_at = datetime.now()

        # Perform trend analysis
        trends = await trend_service.analyze_trends(
            platforms=request.platforms,
            region=request.region,
            age_range=request.age_range,
            limit=request.limit,
            include_hashtags=request.include_hashtags,
            include_sounds=request.include_sounds,
            include_formats=request.include_formats
        )

        # Check if trends were found
        if not trends or len(trends) == 0:
            task_store[task_id].status = "failed"
            task_store[task_id].error_message = "No trends found or API returned empty results"
            task_store[task_id].updated_at = datetime.now()
            return

        # Save results to file
        output_path = f"social_media_api/output/reports/trends_{task_id}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(trends, f, ensure_ascii=False, indent=2)

        # Update task status
        task_store[task_id].status = "completed"
        task_store[task_id].progress = 1.0
        task_store[task_id].result = {
            "trends": trends,
            "report_file": output_path
        }
        task_store[task_id].updated_at = datetime.now()

    except Exception as e:
        # Handle errors
        task_store[task_id].status = "failed"
        task_store[task_id].error_message = str(e)
        task_store[task_id].updated_at = datetime.now()
        logging.error(f"Trend analysis failed: {e}")


@app.post("/video/generate", tags=["Video"])
async def generate_video(
    background_tasks: BackgroundTasks,
    request: VideoGenerationRequest,
    trend_service: TrendAnalysisService = Depends(get_trend_service),
    nlp_service: NLPService = Depends(get_nlp_service),
    video_service: VideoGenerationService = Depends(get_video_service)
):
    """Generate a social media video based on trend analysis"""

    # Generate task ID
    task_id = str(uuid.uuid4())

    # Initialize task status
    task_store[task_id] = TaskStatus(
        task_id=task_id,
        status="processing",
        progress=0.0,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )

    # Start background task
    background_tasks.add_task(
        process_video_generation,
        task_id,
        request,
        trend_service,
        nlp_service,
        video_service
    )

    return {
        "task_id": task_id,
        "status": "processing",
        "message": "Video generation started"
    }


async def process_video_generation(
    task_id: str,
    request: VideoGenerationRequest,
    trend_service: TrendAnalysisService,
    nlp_service: NLPService,
    video_service: VideoGenerationService
):
    """Process video generation in background"""
    try:
        # Update task status
        task_store[task_id].progress = 0.1
        task_store[task_id].updated_at = datetime.now()

        # Get trend data
        trend_data = await trend_service.get_trend_by_id(request.trend_id)

        # Check if trend data exists
        if not trend_data:
            task_store[task_id].status = "failed"
            task_store[task_id].error_message = f"Trend with ID {request.trend_id} not found"
            task_store[task_id].updated_at = datetime.now()
            return

        # Update progress
        task_store[task_id].progress = 0.2
        task_store[task_id].updated_at = datetime.now()

        # Perform NLP analysis on trend data
        nlp_analysis = await nlp_service.analyze_trend(trend_data)

        # Check if NLP analysis was successful
        if not nlp_analysis:
            task_store[task_id].status = "failed"
            task_store[task_id].error_message = "NLP analysis failed or returned empty results"
            task_store[task_id].updated_at = datetime.now()
            return

        # Update progress
        task_store[task_id].progress = 0.4
        task_store[task_id].updated_at = datetime.now()

        # Generate video
        video_result = await video_service.generate_video(
            trend_data=trend_data,
            nlp_analysis=nlp_analysis,
            style=request.style,
            resolution=request.resolution,
            duration=request.duration,
            include_music=request.include_music,
            include_voiceover=request.include_voiceover,
            language=request.language
        )

        # Check if video generation was successful
        if not video_result or "video_path" not in video_result:
            task_store[task_id].status = "failed"
            task_store[task_id].error_message = "Video generation failed or returned empty results"
            task_store[task_id].updated_at = datetime.now()
            return

        # Update task status
        task_store[task_id].status = "completed"
        task_store[task_id].progress = 1.0
        task_store[task_id].result = video_result
        task_store[task_id].updated_at = datetime.now()

    except Exception as e:
        # Handle errors
        task_store[task_id].status = "failed"
        task_store[task_id].error_message = str(e)
        task_store[task_id].updated_at = datetime.now()
        logging.error(f"Video generation failed: {e}")


@app.post("/post/generate", tags=["Post"])
async def generate_post_text(
    background_tasks: BackgroundTasks,
    request: PostTextRequest,
    trend_service: TrendAnalysisService = Depends(get_trend_service),
    post_service: PostGenerationService = Depends(get_post_service)
):
    """Generate post text for a video"""

    # Generate task ID
    task_id = str(uuid.uuid4())

    # Initialize task status
    task_store[task_id] = TaskStatus(
        task_id=task_id,
        status="processing",
        progress=0.0,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )

    # Start background task
    background_tasks.add_task(
        process_post_generation,
        task_id,
        request,
        trend_service,
        post_service
    )

    return {
        "task_id": task_id,
        "status": "processing",
        "message": "Post text generation started"
    }


async def process_post_generation(
    task_id: str,
    request: PostTextRequest,
    trend_service: TrendAnalysisService,
    post_service: PostGenerationService
):
    """Process post text generation in background"""
    try:
        # Update task status
        task_store[task_id].progress = 0.1
        task_store[task_id].updated_at = datetime.now()

        # Get trend data
        trend_data = await trend_service.get_trend_by_id(request.trend_id)

        # Check if trend data exists
        if not trend_data:
            task_store[task_id].status = "failed"
            task_store[task_id].error_message = f"Trend with ID {request.trend_id} not found"
            task_store[task_id].updated_at = datetime.now()
            return

        # Update progress
        task_store[task_id].progress = 0.3
        task_store[task_id].updated_at = datetime.now()

        # Generate post text
        post_result = await post_service.generate_post_text(
            trend_data=trend_data,
            video_id=request.video_id,
            style=request.style,
            hashtag_count=request.hashtag_count,
            include_emojis=request.include_emojis
        )

        # Check if post generation was successful
        if not post_result:
            task_store[task_id].status = "failed"
            task_store[task_id].error_message = "Post text generation failed or returned empty results"
            task_store[task_id].updated_at = datetime.now()
            return

        # Update task status
        task_store[task_id].status = "completed"
        task_store[task_id].progress = 1.0
        task_store[task_id].result = post_result
        task_store[task_id].updated_at = datetime.now()

    except Exception as e:
        # Handle errors
        task_store[task_id].status = "failed"
        task_store[task_id].error_message = str(e)
        task_store[task_id].updated_at = datetime.now()
        logging.error(f"Post text generation failed: {e}")


@app.get("/tasks/{task_id}", response_model=TaskStatus, tags=["Tasks"])
async def get_task_status(task_id: str):
    """Get task status"""
    if task_id not in task_store:
        raise HTTPException(status_code=404, detail="Task not found")

    return task_store[task_id]


@app.get("/tasks", tags=["Tasks"])
async def list_tasks(
    status: Optional[str] = None,
    limit: int = 100
):
    """List all tasks"""
    tasks = list(task_store.values())

    if status:
        tasks = [t for t in tasks if t.status == status]

    # Sort by update time (newest first)
    tasks.sort(key=lambda x: x.updated_at, reverse=True)

    return {
        "tasks": tasks[:limit],
        "total": len(tasks),
        "filtered": status is not None
    }


@app.get("/download/{filename}", tags=["Files"])
async def download_file(filename: str):
    """Download generated file"""
    # Check if file is a video
    if filename.endswith((".mp4", ".mov", ".avi")):
        file_path = f"social_media_api/output/videos/{filename}"
    else:
        file_path = f"social_media_api/output/reports/{filename}"

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='application/octet-stream'
    )


@app.delete("/tasks/{task_id}", tags=["Tasks"])
async def delete_task(task_id: str):
    """Delete task and associated files"""
    if task_id not in task_store:
        raise HTTPException(status_code=404, detail="Task not found")

    # Delete associated files
    task = task_store[task_id]
    if task.result:
        if "video_path" in task.result:
            video_path = task.result["video_path"]
            if os.path.exists(video_path):
                os.remove(video_path)

        if "report_file" in task.result:
            report_path = task.result["report_file"]
            if os.path.exists(report_path):
                os.remove(report_path)

    # Remove task from store
    del task_store[task_id]

    return {"message": "Task deleted successfully"}


@app.get("/stats", tags=["General"])
async def get_statistics():
    """API statistics"""
    total_tasks = len(task_store)
    completed = len([t for t in task_store.values()
                    if t.status == "completed"])
    failed = len([t for t in task_store.values() if t.status == "failed"])
    processing = len([t for t in task_store.values()
                     if t.status == "processing"])

    return {
        "total_tasks": total_tasks,
        "completed": completed,
        "failed": failed,
        "processing": processing,
        "success_rate": completed / max(total_tasks, 1) * 100,
        "disk_usage": get_disk_usage(),
        "uptime": "TODO: Implement uptime tracking"
    }


def get_disk_usage():
    """Get disk usage"""
    try:
        total, used, free = shutil.disk_usage("social_media_api/output/")
        return {
            "total_gb": total / (1024**3),
            "used_gb": used / (1024**3),
            "free_gb": free / (1024**3),
            "usage_percent": (used / total) * 100
        }
    except:
        return {"error": "Could not determine disk usage"}


@app.on_event("startup")
async def setup_scheduled_task():
    """Set up scheduled task for daily video generation"""
    async def run_daily_generation():
        """Run daily video generation at 21:30"""
        while True:
            now = datetime.now()
            # Wait until 21:30
            target_time = now.replace(
                hour=21, minute=30, second=0, microsecond=0)
            if now >= target_time:
                target_time = target_time.replace(day=target_time.day + 1)

            # Calculate seconds until target time
            wait_seconds = (target_time - now).total_seconds()
            await asyncio.sleep(wait_seconds)

            # Run the daily generation
            try:
                logging.info("Starting scheduled daily video generation")

                # Analyze trends
                trend_request = TrendAnalysisRequest()
                trend_result = await trend_service.analyze_trends(
                    platforms=trend_request.platforms,
                    region=trend_request.region,
                    age_range=trend_request.age_range,
                    limit=trend_request.limit,
                    include_hashtags=trend_request.include_hashtags,
                    include_sounds=trend_request.include_sounds,
                    include_formats=trend_request.include_formats
                )

                if not trend_result or len(trend_result) == 0:
                    logging.error("Scheduled task: No trends found")
                    continue

                # Select top trend
                top_trend = trend_result[0]
                trend_id = top_trend.get("id", str(uuid.uuid4()))

                # Perform NLP analysis
                nlp_analysis = await nlp_service.analyze_trend(top_trend)

                if not nlp_analysis:
                    logging.error("Scheduled task: NLP analysis failed")
                    continue

                # Generate video
                video_request = VideoGenerationRequest(trend_id=trend_id)
                video_result = await video_service.generate_video(
                    trend_data=top_trend,
                    nlp_analysis=nlp_analysis,
                    style=video_request.style,
                    resolution=video_request.resolution,
                    duration=video_request.duration,
                    include_music=video_request.include_music,
                    include_voiceover=video_request.include_voiceover,
                    language=video_request.language
                )

                if not video_result or "video_path" not in video_result:
                    logging.error("Scheduled task: Video generation failed")
                    continue

                # Generate post text
                video_id = video_result.get("video_id", str(uuid.uuid4()))
                post_request = PostTextRequest(
                    video_id=video_id, trend_id=trend_id)
                post_result = await post_service.generate_post_text(
                    trend_data=top_trend,
                    video_id=video_id,
                    style=post_request.style,
                    hashtag_count=post_request.hashtag_count,
                    include_emojis=post_request.include_emojis
                )

                if not post_result:
                    logging.error(
                        "Scheduled task: Post text generation failed")
                    continue

                # Log success
                logging.info(
                    f"Scheduled task completed successfully. Video: {video_result.get('video_path')}")

                # Save report
                report = {
                    "timestamp": datetime.now().isoformat(),
                    "trend": top_trend,
                    "nlp_analysis": nlp_analysis,
                    "video": video_result,
                    "post": post_result
                }

                report_path = f"social_media_api/output/reports/daily_report_{datetime.now().strftime('%Y%m%d')}.json"
                with open(report_path, 'w', encoding='utf-8') as f:
                    json.dump(report, f, ensure_ascii=False, indent=2)

            except Exception as e:
                logging.error(f"Scheduled task failed: {e}")

    # Start the scheduled task
    asyncio.create_task(run_daily_generation())


# Serve static files for the UI
app.mount(
    "/ui", StaticFiles(directory="social_media_api/app/ui", html=True), name="ui")


if __name__ == "__main__":
    uvicorn.run(
        "social_media_api.api_server.social_media_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
