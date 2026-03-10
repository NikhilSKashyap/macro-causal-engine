"""
main.py
-------
FastAPI production serving layer for the Macro Causal Engine.

Exposes a single high-level endpoint that accepts raw macroeconomic text,
runs it through the full extraction + sequencification pipeline, and returns
a structured response containing both the typed Pydantic DAG and the
PyTorch-ready text sequence.

Endpoints
~~~~~~~~~
GET  /health                  — liveness / readiness probe
POST /extract-and-sequencify  — full pipeline: text → DAG → sequence

Architecture notes
~~~~~~~~~~~~~~~~~~
- The Anthropic client call (extractor.extract_causal_dag) is synchronous.
  Running it directly in an async handler would block the event loop and
  starve all other concurrent requests.  We offload it to a thread-pool
  executor via asyncio.to_thread() so FastAPI's event loop stays free.

- A single tokenizer instance is created at startup (lifespan) and stored
  in app.state to avoid re-initialising the HF vocab on every request.

- Request IDs (UUID4) are injected via middleware so every log line and
  response payload can be correlated across services.

- Detailed latency metadata is returned in every response, broken down by
  pipeline stage, enabling SLA monitoring without an external tracer.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from extractor import CausalDAG, extract_causal_dag
from sequencifier import load_tokenizer, sequencify_dag

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialise shared resources on startup; release them on shutdown.

    Resources created here:
    - Tokenizer: loaded once and stored in app.state.tokenizer.
      Re-using a single instance avoids repeated vocab deserialization and
      keeps memory flat under concurrent load.
    """
    logger.info("Starting up Macro Causal Engine …")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning(
            "ANTHROPIC_API_KEY not set — /extract-and-sequencify will fail at runtime."
        )

    logger.info("Loading tokenizer …")
    app.state.tokenizer = load_tokenizer(max_length=512)
    logger.info("Tokenizer ready: %s", type(app.state.tokenizer).__name__)

    yield  # ── application running ──────────────────────────────────────────

    logger.info("Shutting down Macro Causal Engine.")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Macro Causal Engine",
    description=(
        "Production API for extracting Causal DAGs from macroeconomic text "
        "and converting them to PyTorch-ready autoregressive sequences."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request-ID middleware
# ---------------------------------------------------------------------------


@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    """
    Inject a UUID4 request ID into every request/response cycle.

    The ID is appended to the response header ``X-Request-ID`` and is also
    stored on the request state so endpoint handlers can embed it in their
    response payload for end-to-end traceability.
    """
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ---------------------------------------------------------------------------
# Pydantic I/O models
# ---------------------------------------------------------------------------


class ExtractionRequest(BaseModel):
    """
    Payload for POST /extract-and-sequencify.

    Attributes
    ----------
    text : str
        Raw macroeconomic prose (e.g. FOMC minutes excerpt, BIS paper paragraph).
        Must be at least 50 characters to be meaningful.
    model : str
        Anthropic model to use for extraction.
        Defaults to ``claude-sonnet-4-5``.
    """

    text: str = Field(
        ...,
        min_length=50,
        description="Raw macroeconomic text to extract a causal DAG from.",
        examples=[
            "The persistent inversion of the Treasury yield curve suppressed net "
            "interest margins at regional banks, inducing a meaningful tightening "
            "of commercial and industrial lending standards …"
        ],
    )
    model: str = Field(
        default="claude-sonnet-4-5",
        description="Anthropic model ID to use for causal extraction.",
    )


class PipelineMetadata(BaseModel):
    """Latency and structural metadata for a single pipeline run."""

    request_id: str
    model: str
    node_count: int
    edge_count: int
    confounder_count: int
    sequence_char_length: int
    extraction_latency_ms: float = Field(
        ..., description="Time spent calling the Anthropic API, in milliseconds."
    )
    sequencify_latency_ms: float = Field(
        ..., description="Time spent in the sequencifier, in milliseconds."
    )
    total_latency_ms: float = Field(
        ..., description="End-to-end wall-clock time, in milliseconds."
    )


class ExtractionResponse(BaseModel):
    """
    Response payload for POST /extract-and-sequencify.

    Attributes
    ----------
    dag : dict[str, Any]
        The full CausalDAG serialised as a JSON-compatible dict.
        Consumers can reconstruct the Pydantic model via
        ``CausalDAG.model_validate(response.dag)``.
    sequence : str
        The topologically sorted, tokenizer-ready text sequence produced by
        sequencifier.sequencify_dag().
    metadata : PipelineMetadata
        Per-request latency breakdown and graph statistics.
    """

    dag: dict[str, Any]
    sequence: str
    metadata: PipelineMetadata


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get(
    "/health",
    summary="Liveness / readiness probe",
    tags=["ops"],
)
async def health(request: Request) -> JSONResponse:
    """
    Returns 200 OK when the service is alive and the tokenizer is loaded.

    Suitable for use as a Kubernetes liveness and readiness probe.
    """
    tokenizer_ready = hasattr(request.app.state, "tokenizer")
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "ok",
            "tokenizer": type(request.app.state.tokenizer).__name__ if tokenizer_ready else "not loaded",
            "api_key_configured": bool(os.environ.get("ANTHROPIC_API_KEY")),
        },
    )


@app.post(
    "/extract-and-sequencify",
    response_model=ExtractionResponse,
    summary="Extract a causal DAG and sequencify it for autoregressive modelling",
    tags=["pipeline"],
    status_code=status.HTTP_200_OK,
)
async def extract_and_sequencify(
    payload: ExtractionRequest,
    request: Request,
) -> ExtractionResponse:
    """
    Full pipeline: raw macro text → CausalDAG → PyTorch-ready text sequence.

    Pipeline stages
    ~~~~~~~~~~~~~~~
    1. **Extraction** (I/O-bound, ~2–4 s): calls Claude via Anthropic tool-use
       API in a thread-pool executor to avoid blocking the event loop.
    2. **Sequencification** (CPU-bound, <5 ms): topological sort + micro-grammar
       serialisation; runs synchronously since it is near-instantaneous.

    Returns
    -------
    ExtractionResponse
        ``dag``      — full typed DAG as a JSON dict.
        ``sequence`` — structured text sequence ready for tokenization.
        ``metadata`` — per-stage latency and graph statistics.

    Raises
    ------
    422 Unprocessable Entity
        If the request body fails Pydantic validation (e.g. text too short).
    503 Service Unavailable
        If the Anthropic API is unreachable or returns an error.
    500 Internal Server Error
        If the extracted payload fails DAG schema validation.
    """
    request_id: str = request.state.request_id
    t_total_start = time.perf_counter()

    logger.info(
        "request_id=%s | model=%s | text_len=%d",
        request_id,
        payload.model,
        len(payload.text),
    )

    # ── Stage 1: Extraction (offloaded to thread pool) ────────────────────
    t_extract_start = time.perf_counter()
    try:
        dag: CausalDAG = await asyncio.to_thread(
            extract_causal_dag,
            payload.text,
            model=payload.model,
        )
    except Exception as exc:
        logger.exception("request_id=%s | Extraction failed: %s", request_id, exc)
        # Distinguish API errors from schema validation errors
        if "tool" in str(exc).lower() or "api" in type(exc).__name__.lower():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Anthropic API error: {exc}",
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"DAG extraction failed: {exc}",
        ) from exc

    extraction_ms = (time.perf_counter() - t_extract_start) * 1000

    logger.info(
        "request_id=%s | extracted %d nodes, %d edges in %.1f ms",
        request_id,
        len(dag.nodes),
        len(dag.edges),
        extraction_ms,
    )

    # ── Stage 2: Sequencification (synchronous, sub-millisecond) ─────────
    t_seq_start = time.perf_counter()
    try:
        sequence: str = sequencify_dag(dag)
    except Exception as exc:
        logger.exception("request_id=%s | Sequencification failed: %s", request_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sequencification failed: {exc}",
        ) from exc

    sequencify_ms = (time.perf_counter() - t_seq_start) * 1000
    total_ms = (time.perf_counter() - t_total_start) * 1000

    confounder_count = sum(1 for e in dag.edges if e.confounder is not None)

    logger.info(
        "request_id=%s | seq_len=%d chars | total=%.1f ms",
        request_id,
        len(sequence),
        total_ms,
    )

    return ExtractionResponse(
        dag=dag.model_dump(),
        sequence=sequence,
        metadata=PipelineMetadata(
            request_id=request_id,
            model=payload.model,
            node_count=len(dag.nodes),
            edge_count=len(dag.edges),
            confounder_count=confounder_count,
            sequence_char_length=len(sequence),
            extraction_latency_ms=round(extraction_ms, 2),
            sequencify_latency_ms=round(sequencify_ms, 2),
            total_latency_ms=round(total_ms, 2),
        ),
    )


# ---------------------------------------------------------------------------
# Global exception handlers
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all for any unhandled exception; prevents raw tracebacks leaking."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.exception("request_id=%s | Unhandled exception: %s", request_id, exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error.", "request_id": request_id},
    )


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        workers=1,  # single worker; scale horizontally behind a load balancer
    )
