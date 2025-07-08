#!/usr/bin/env python3
"""
🚀 Trading Card Optimization API Server
FastAPI-basierte REST-Schnittstelle für automatisierte Trading Card Optimierung
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
from pydantic import BaseModel, Field
import uvicorn
from PIL import Image
import io

# Import unserer Trading Card Optimizer
from trading_card_optimizer import TradingCardOptimizer, OptimizationSettings, TradingCardAPI

# API Models


class OptimizationRequest(BaseModel):
    """Request-Model für Trading Card Optimierung"""
    target_resolution: Optional[tuple] = (512, 768)
    upscale_factor: int = Field(default=2, ge=1, le=4)
    edge_enhancement: float = Field(default=1.5, ge=0.5, le=3.0)
    text_sharpening: float = Field(default=2.0, ge=0.5, le=4.0)
    color_saturation: float = Field(default=1.1, ge=0.5, le=2.0)
    contrast_boost: float = Field(default=1.2, ge=0.5, le=2.0)
    noise_reduction: float = Field(default=0.3, ge=0.0, le=1.0)
    style_strength: float = Field(default=0.8, ge=0.0, le=1.0)
    ocr_languages: List[str] = ["en", "de"]
    apply_style_transfer: bool = True
    validation_threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class BatchRequest(BaseModel):
    """Request-Model für Batch-Processing"""
    optimization_settings: OptimizationRequest
    notification_webhook: Optional[str] = None
    priority: str = Field(default="normal", regex="^(low|normal|high)$")


class OptimizationResponse(BaseModel):
    """Response-Model für Trading Card Optimierung"""
    success: bool
    task_id: str
    processing_time: float
    original_filename: str
    enhanced_filename: str
    quality_score: float
    analysis_data: Dict[str, Any]
    enhancement_report: Dict[str, Any]
    validation_report: Dict[str, Any]


class BatchResponse(BaseModel):
    """Response-Model für Batch-Processing"""
    success: bool
    batch_id: str
    total_images: int
    processed_images: int
    failed_images: int
    processing_time: float
    results: List[OptimizationResponse]


class TaskStatus(BaseModel):
    """Task-Status Model"""
    task_id: str
    status: str  # pending, processing, completed, failed
    progress: float  # 0.0 - 1.0
    created_at: datetime
    updated_at: datetime
    result: Optional[OptimizationResponse] = None
    error_message: Optional[str] = None


# Global state management
task_store = {}
app = FastAPI(
    title="Trading Card Optimizer API",
    description="🎮 API für die automatisierte Optimierung von Trading Cards mit KI-gestützter Bildverbesserung",
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

# Global optimizer instance
optimizer = None
trading_card_api = None


def get_optimizer():
    """Optimizer Dependency"""
    global optimizer, trading_card_api
    if optimizer is None:
        optimizer = TradingCardOptimizer()
        trading_card_api = TradingCardAPI(optimizer)
    return trading_card_api


@app.on_event("startup")
async def startup_event():
    """Server-Initialisierung"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Trading Card Optimizer API starting up...")

    # Output-Verzeichnisse erstellen
    os.makedirs("output/enhanced", exist_ok=True)
    os.makedirs("output/analysis", exist_ok=True)
    os.makedirs("output/reports", exist_ok=True)
    os.makedirs("temp", exist_ok=True)

    # Optimizer initialisieren
    get_optimizer()
    logger.info("API ready!")


@app.get("/", tags=["General"])
async def root():
    """API-Status und Informationen"""
    return {
        "message": "🎮 Trading Card Optimizer API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "optimize_single": "/optimize/single",
            "optimize_batch": "/optimize/batch",
            "task_status": "/tasks/{task_id}",
            "download": "/download/{filename}"
        },
        "docs": "/docs"
    }


@app.get("/health", tags=["General"])
async def health_check():
    """Health Check Endpoint"""
    try:
        # GPU-Status prüfen
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_memory = torch.cuda.get_device_properties(
            0).total_memory if gpu_available else 0

        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "gpu_available": gpu_available,
            "gpu_memory_gb": gpu_memory / (1024**3) if gpu_memory > 0 else 0,
            "active_tasks": len([t for t in task_store.values() if t.status == "processing"])
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Health check failed: {str(e)}")


@app.post("/optimize/single", response_model=OptimizationResponse, tags=["Optimization"])
async def optimize_single_card(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    settings: OptimizationRequest = Depends(),
    api: TradingCardAPI = Depends(get_optimizer)
):
    """Einzelne Trading Card optimieren"""

    # Task-ID generieren
    task_id = str(uuid.uuid4())

    # Task-Status initialisieren
    task_store[task_id] = TaskStatus(
        task_id=task_id,
        status="processing",
        progress=0.0,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )

    try:
        # Datei validieren
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400, detail="File must be an image")

        # Bild laden
        image_data = await file.read()

        # Optimierung-Settings konvertieren
        opt_settings = OptimizationSettings(
            target_resolution=settings.target_resolution,
            upscale_factor=settings.upscale_factor,
            edge_enhancement=settings.edge_enhancement,
            text_sharpening=settings.text_sharpening,
            style_strength=settings.style_strength
        )

        # Progress Update
        task_store[task_id].progress = 0.2
        task_store[task_id].updated_at = datetime.now()

        # Optimierung durchführen
        start_time = datetime.now()
        result = await api.optimize_single_card(image_data, opt_settings)
        processing_time = (datetime.now() - start_time).total_seconds()

        # Ergebnis speichern
        output_filename = f"enhanced_{task_id}_{file.filename}"
        output_path = f"output/enhanced/{output_filename}"

        with open(output_path, 'wb') as f:
            f.write(result["enhanced_image"])

        # Response erstellen
        response = OptimizationResponse(
            success=result["success"],
            task_id=task_id,
            processing_time=processing_time,
            original_filename=file.filename,
            enhanced_filename=output_filename,
            quality_score=result["quality_metrics"]["overall_score"],
            analysis_data=result["analysis"],
            enhancement_report={"processing_time": processing_time},
            validation_report=result["quality_metrics"]
        )

        # Task abschließen
        task_store[task_id].status = "completed"
        task_store[task_id].progress = 1.0
        task_store[task_id].result = response
        task_store[task_id].updated_at = datetime.now()

        return response

    except Exception as e:
        # Error handling
        task_store[task_id].status = "failed"
        task_store[task_id].error_message = str(e)
        task_store[task_id].updated_at = datetime.now()

        raise HTTPException(
            status_code=500, detail=f"Optimization failed: {str(e)}")


@app.post("/optimize/batch", response_model=Dict[str, str], tags=["Optimization"])
async def optimize_batch(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    settings: BatchRequest = Depends(),
    api: TradingCardAPI = Depends(get_optimizer)
):
    """Batch-Optimierung für mehrere Trading Cards"""

    # Batch-ID generieren
    batch_id = str(uuid.uuid4())

    if len(files) > 50:
        raise HTTPException(
            status_code=400, detail="Maximum 50 files per batch")

    # Batch-Task im Hintergrund starten
    background_tasks.add_task(
        process_batch_async,
        batch_id,
        files,
        settings,
        api
    )

    return {
        "batch_id": batch_id,
        "status": "processing",
        "total_files": len(files),
        "message": f"Batch processing started with {len(files)} files"
    }


async def process_batch_async(
    batch_id: str,
    files: List[UploadFile],
    settings: BatchRequest,
    api: TradingCardAPI
):
    """Asynchrone Batch-Verarbeitung"""

    # Batch-Status initialisieren
    task_store[batch_id] = TaskStatus(
        task_id=batch_id,
        status="processing",
        progress=0.0,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )

    results = []
    failed_count = 0

    try:
        start_time = datetime.now()

        # Optimierung-Settings konvertieren
        opt_settings = OptimizationSettings(
            target_resolution=settings.optimization_settings.target_resolution,
            upscale_factor=settings.optimization_settings.upscale_factor,
            edge_enhancement=settings.optimization_settings.edge_enhancement,
            text_sharpening=settings.optimization_settings.text_sharpening
        )

        # Dateien einzeln verarbeiten
        for i, file in enumerate(files):
            try:
                # Progress Update
                progress = (i + 1) / len(files)
                task_store[batch_id].progress = progress
                task_store[batch_id].updated_at = datetime.now()

                # Einzelne Datei verarbeiten
                image_data = await file.read()
                result = await api.optimize_single_card(image_data, opt_settings)

                # Ergebnis speichern
                output_filename = f"batch_{batch_id}_{i}_{file.filename}"
                output_path = f"output/enhanced/{output_filename}"

                with open(output_path, 'wb') as f:
                    f.write(result["enhanced_image"])

                # Response für diese Datei
                file_response = OptimizationResponse(
                    success=result["success"],
                    task_id=f"{batch_id}_{i}",
                    processing_time=0.0,  # Individual timing not tracked in batch
                    original_filename=file.filename,
                    enhanced_filename=output_filename,
                    quality_score=result["quality_metrics"]["overall_score"],
                    analysis_data=result["analysis"],
                    enhancement_report={},
                    validation_report=result["quality_metrics"]
                )

                results.append(file_response)

            except Exception as e:
                failed_count += 1
                logging.error(f"Failed to process file {file.filename}: {e}")
                continue

        # Batch abschließen
        processing_time = (datetime.now() - start_time).total_seconds()

        batch_response = BatchResponse(
            success=True,
            batch_id=batch_id,
            total_images=len(files),
            processed_images=len(results),
            failed_images=failed_count,
            processing_time=processing_time,
            results=results
        )

        task_store[batch_id].status = "completed"
        task_store[batch_id].progress = 1.0
        task_store[batch_id].result = batch_response
        task_store[batch_id].updated_at = datetime.now()

        # Webhook-Benachrichtigung (falls konfiguriert)
        if settings.notification_webhook:
            # TODO: Webhook-Implementierung
            pass

    except Exception as e:
        # Batch-Fehler
        task_store[batch_id].status = "failed"
        task_store[batch_id].error_message = str(e)
        task_store[batch_id].updated_at = datetime.now()
        logging.error(f"Batch processing failed: {e}")


@app.get("/tasks/{task_id}", response_model=TaskStatus, tags=["Tasks"])
async def get_task_status(task_id: str):
    """Task-Status abfragen"""
    if task_id not in task_store:
        raise HTTPException(status_code=404, detail="Task not found")

    return task_store[task_id]


@app.get("/tasks", tags=["Tasks"])
async def list_tasks(
    status: Optional[str] = None,
    limit: int = 100
):
    """Alle Tasks auflisten"""
    tasks = list(task_store.values())

    if status:
        tasks = [t for t in tasks if t.status == status]

    # Nach Update-Zeit sortieren (neueste zuerst)
    tasks.sort(key=lambda x: x.updated_at, reverse=True)

    return {
        "tasks": tasks[:limit],
        "total": len(tasks),
        "filtered": status is not None
    }


@app.get("/download/{filename}", tags=["Files"])
async def download_file(filename: str):
    """Optimierte Datei herunterladen"""
    file_path = f"output/enhanced/{filename}"

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='application/octet-stream'
    )


@app.delete("/tasks/{task_id}", tags=["Tasks"])
async def delete_task(task_id: str):
    """Task und zugehörige Dateien löschen"""
    if task_id not in task_store:
        raise HTTPException(status_code=404, detail="Task not found")

    # Zugehörige Dateien löschen
    task = task_store[task_id]
    if task.result and hasattr(task.result, 'enhanced_filename'):
        file_path = f"output/enhanced/{task.result.enhanced_filename}"
        if os.path.exists(file_path):
            os.remove(file_path)

    # Task aus Store entfernen
    del task_store[task_id]

    return {"message": "Task deleted successfully"}


@app.get("/stats", tags=["General"])
async def get_statistics():
    """API-Statistiken"""
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
    """Festplattenspeicher-Nutzung ermitteln"""
    try:
        total, used, free = shutil.disk_usage("output/")
        return {
            "total_gb": total / (1024**3),
            "used_gb": used / (1024**3),
            "free_gb": free / (1024**3),
            "usage_percent": (used / total) * 100
        }
    except:
        return {"error": "Could not determine disk usage"}

# Cleanup-Task für alte Dateien


@app.on_event("startup")
async def setup_cleanup_task():
    """Cleanup-Task für alte Dateien einrichten"""
    async def cleanup_old_files():
        while True:
            try:
                # Alle 6 Stunden ausführen
                await asyncio.sleep(6 * 60 * 60)

                # Dateien älter als 24h löschen
                cutoff_time = datetime.now().timestamp() - (24 * 60 * 60)

                for file_path in Path("output/enhanced").glob("*"):
                    if file_path.stat().st_ctime < cutoff_time:
                        file_path.unlink()
                        logging.info(f"Cleaned up old file: {file_path}")

                # Alte Tasks aus Store entfernen
                old_task_ids = [
                    task_id for task_id, task in task_store.items()
                    if task.updated_at.timestamp() < cutoff_time
                ]

                for task_id in old_task_ids:
                    del task_store[task_id]

                logging.info(
                    f"Cleanup completed. Removed {len(old_task_ids)} old tasks")

            except Exception as e:
                logging.error(f"Cleanup error: {e}")

    # Cleanup-Task im Hintergrund starten
    asyncio.create_task(cleanup_old_files())

if __name__ == "__main__":
    uvicorn.run(
        "trading_card_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
