"""
FastAPI Server for Legal Document Summarization
Provides REST API for model inference with rate limiting and monitoring.
"""

import os
import time
import logging
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from api.inference import InferenceEngine
from server.model_registry import ModelRegistry


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="Secure Legal FL API",
    description="Federated Learning API for Legal Document Summarization",
    version="1.0.0"
)

# Add rate limiting middleware
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware (configure for production)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Security
security = HTTPBearer(auto_error=False)


# Global variables
inference_engine: Optional[InferenceEngine] = None
model_registry: Optional[ModelRegistry] = None


class SummarizationRequest(BaseModel):
    """Request model for summarization."""
    text: str = Field(..., min_length=10, max_length=10000,
                     description="Legal document text to summarize")
    max_length: Optional[int] = Field(512, ge=50, le=1024,
                                    description="Maximum summary length")
    temperature: Optional[float] = Field(0.7, ge=0.1, le=2.0,
                                       description="Sampling temperature")


class SummarizationResponse(BaseModel):
    """Response model for summarization."""
    summary: str = Field(..., description="Generated summary")
    model_version: str = Field(..., description="Model version used")
    processing_time: float = Field(..., description="Processing time in seconds")
    metrics: Dict[str, Any] = Field(..., description="Model performance metrics")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    current_version: Optional[str] = Field(None, description="Current model version")
    uptime: float = Field(..., description="Service uptime in seconds")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global inference_engine, model_registry

    # Startup
    logger.info("Starting Secure Legal FL API...")

    try:
        # Initialize model registry
        model_registry = ModelRegistry()

        # Initialize inference engine with current model
        current_model = model_registry.get_current_model()
        if current_model:
            model_path = current_model["path"]
            inference_engine = InferenceEngine(model_path)
            logger.info(f"Loaded model version: {model_registry.metadata.get('current_version')}")
        else:
            logger.warning("No deployed model found. API will return errors until model is deployed.")

    except Exception as e:
        logger.error(f"Failed to initialize API: {e}")
        # Continue without model - API will handle gracefully

    yield

    # Shutdown
    logger.info("Shutting down Secure Legal FL API...")
    if inference_engine:
        inference_engine.unload_model()


# Initialize FastAPI with lifespan
app = FastAPI(lifespan=lifespan)


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> bool:
    """
    Verify authentication token.
    In production, implement proper token validation.
    """
    if not credentials:
        return False

    # Placeholder token validation
    # In production, validate against a proper auth service
    valid_tokens = os.getenv("API_TOKENS", "").split(",")
    return credentials.credentials in valid_tokens


@app.get("/health", response_model=HealthResponse)
@limiter.limit("30/minute")
async def health_check(request: Request):
    """Health check endpoint."""
    model_loaded = inference_engine is not None and inference_engine.model is not None
    current_version = None

    if model_registry:
        current_version = model_registry.metadata.get("current_version")

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        current_version=current_version,
        uptime=time.time() - request.app.state.start_time if hasattr(request.app, 'state') else 0
    )


@app.post("/summarize", response_model=SummarizationResponse)
@limiter.limit("10/minute")
async def summarize_document(
    request: SummarizationRequest,
    req: Request,
    authenticated: bool = Depends(verify_token)
):
    """
    Summarize a legal document.

    Requires authentication token in production.
    """
    if not authenticated:
        raise HTTPException(status_code=401, detail="Authentication required")

    if not inference_engine or not inference_engine.model:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Please try again later."
        )

    try:
        start_time = time.time()

        # Generate summary
        result = inference_engine.generate_summary(
            text=request.text,
            max_length=request.max_length,
            temperature=request.temperature
        )

        processing_time = time.time() - start_time

        # Get model version
        model_version = "unknown"
        if model_registry:
            current = model_registry.get_current_model()
            if current:
                model_version = model_registry.metadata.get("current_version", "unknown")

        # Get model metrics
        metrics = {}
        if model_registry and model_version != "unknown":
            model_info = model_registry.get_model_info(model_version)
            if model_info:
                metrics = model_info.get("metrics", {})

        logger.info(f"Generated summary in {processing_time:.2f}s for text length {len(request.text)}")

        return SummarizationResponse(
            summary=result["summary"],
            model_version=model_version,
            processing_time=processing_time,
            metrics=metrics
        )

    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/metrics")
@limiter.limit("10/minute")
async def get_model_metrics(
    req: Request,
    authenticated: bool = Depends(verify_token)
):
    """Get current model performance metrics."""
    if not authenticated:
        raise HTTPException(status_code=401, detail="Authentication required")

    if not model_registry:
        raise HTTPException(status_code=503, detail="Model registry not available")

    try:
        current_model = model_registry.get_current_model()
        if not current_model:
            return {"message": "No model currently deployed"}

        metrics_history = model_registry.get_metrics_history()

        # Get latest metrics
        if not metrics_history.empty:
            latest_metrics = metrics_history.iloc[-1].to_dict()
        else:
            latest_metrics = {}

        # Get performance trend
        trend_analysis = model_registry.get_performance_trend()

        return {
            "current_model": current_model,
            "latest_metrics": latest_metrics,
            "performance_trend": trend_analysis,
            "total_versions": len(model_registry.list_versions())
        }

    except Exception as e:
        logger.error(f"Error retrieving metrics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/versions")
@limiter.limit("5/minute")
async def list_model_versions(
    req: Request,
    authenticated: bool = Depends(verify_token)
):
    """List all model versions."""
    if not authenticated:
        raise HTTPException(status_code=401, detail="Authentication required")

    if not model_registry:
        raise HTTPException(status_code=503, detail="Model registry not available")

    try:
        versions = model_registry.list_versions()
        return {"versions": versions}

    except Exception as e:
        logger.error(f"Error listing versions: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/reload-model")
@limiter.limit("2/minute")
async def reload_model(
    req: Request,
    authenticated: bool = Depends(verify_token)
):
    """Reload the current model (admin operation)."""
    if not authenticated:
        raise HTTPException(status_code=401, detail="Authentication required")

    if not model_registry:
        raise HTTPException(status_code=503, detail="Model registry not available")

    try:
        global inference_engine

        current_model = model_registry.get_current_model()
        if current_model:
            model_path = current_model["path"]
            inference_engine = InferenceEngine(model_path)
            return {"message": f"Model reloaded from {model_path}"}
        else:
            raise HTTPException(status_code=404, detail="No deployed model found")

    except Exception as e:
        logger.error(f"Error reloading model: {e}")
        raise HTTPException(status_code=500, detail="Failed to reload model")


if __name__ == "__main__":
    import uvicorn

    # Set start time for uptime tracking
    app.state.start_time = time.time()

    # Run server
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1  # Increase for production
    )