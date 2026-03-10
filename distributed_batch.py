"""
distributed_batch.py
--------------------
Ray-based distributed batch pipeline for processing 100,000 FOMC documents
in parallel across a multi-node cluster.

This module demonstrates production-grade high-throughput data engineering
patterns using Ray — specifically:

  - @ray.remote functions for stateless, embarrassingly-parallel work
  - @ray.remote Actor classes for stateful coordination (rate limiting,
    result aggregation, dead-letter queuing)
  - ray.wait() for streaming result processing instead of blocking on all futures
  - Resource-aware scheduling (num_cpus, memory) so Ray's scheduler can
    bin-pack tasks optimally across heterogeneous nodes
  - Exponential-backoff retry logic to handle transient Anthropic API errors
  - A three-stage pipeline: Extract → Sequencify → Persist, each as
    independent remote tasks so stages can be independently scaled

Cluster sizing guidance (100k documents)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Anthropic rate limit (Sonnet): ~50 req/s on Tier 3 → ~33 min wall-clock
  at max throughput.  A 20-node cluster with 4 vCPUs each comfortably
  saturates the API with the semaphore-based RateLimiterActor controlling
  concurrent in-flight calls.

  Each extraction task is I/O-bound (~2–4 s RTT); each sequencify task is
  CPU-bound but sub-millisecond.  Ray automatically co-locates sequencify
  tasks on the same node as the completed extraction object reference.

Usage (local simulation)
~~~~~~~~~~~~~~~~~~~~~~~~
    python distributed_batch.py                  # simulate 100 docs locally
    python distributed_batch.py --docs 1000      # simulate 1 000 docs
    RAY_ADDRESS=ray://<head>:10001 python distributed_batch.py --docs 100000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import ray

from extractor import (
    CausalDAG,
    _DAG_TOOL,
    _SYSTEM_PROMPT,
    extract_causal_dag,
)
from sequencifier import sequencify_dag

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------

# Maximum concurrent in-flight Anthropic API calls across the entire cluster.
# Tune this to stay within your Anthropic tier's rate limits.
MAX_CONCURRENT_API_CALLS: int = 40

# Number of Ray futures to submit before waiting for a batch to drain.
# Prevents the Ray object store from being flooded with millions of pending refs.
SUBMIT_BATCH_SIZE: int = 200

# Seconds to wait between exponential-backoff retry attempts.
RETRY_BASE_DELAY_S: float = 1.0
MAX_RETRIES: int = 3

# Default output directory for persisted sequences.
OUTPUT_DIR: Path = Path("output/sequences")

# ---------------------------------------------------------------------------
# Mock document generator
# ---------------------------------------------------------------------------

_FOMC_TEMPLATES: list[str] = [
    (
        "The Committee observed that {indicator} had {direction} by {magnitude} basis points, "
        "reflecting {driver} and contributing to {outcome}. Members noted that {secondary} "
        "remained {condition}, suggesting that {policy_action} would be appropriate given "
        "the dual mandate objectives of maximum employment and price stability. Several "
        "participants highlighted the risk that {risk_factor} could {risk_outcome} if "
        "monetary policy accommodation were maintained beyond the point consistent with "
        "returning inflation sustainably to the 2 percent objective."
    ),
    (
        "Staff projections indicated that {indicator} would {direction} to {magnitude} percent "
        "over the projection horizon, contingent on {driver}. The {secondary} remained "
        "{condition} amid {risk_factor}, while {outcome} continued to reflect the lagged "
        "effects of prior policy tightening. Participants discussed the appropriate pace of "
        "{policy_action}, weighing the risks of {risk_outcome} against the costs of "
        "premature policy normalization in an environment of elevated uncertainty."
    ),
    (
        "Financial conditions {direction} materially, with {indicator} {condition} by "
        "{magnitude} basis points as market participants revised their expectations for "
        "{policy_action}. The {secondary} transmitted {driver} through the {outcome} "
        "channel, compressing risk premia and widening the divergence between {risk_factor} "
        "and fundamentals. Committee members cautioned that {risk_outcome} could amplify "
        "volatility in the event of an unexpected data release or geopolitical shock."
    ),
]

_TEMPLATE_VARS: dict[str, list[str]] = {
    "indicator": [
        "the 2-year Treasury yield", "headline PCE inflation", "core CPI",
        "the federal funds rate", "credit spreads", "the unemployment rate",
        "real GDP growth", "the dollar index", "M2 money supply",
    ],
    "direction": ["risen", "fallen", "tightened", "eased", "compressed", "widened"],
    "magnitude": [str(x) for x in [15, 25, 50, 75, 100, 125, 140, 200]],
    "driver": [
        "OPEC+ supply curtailments", "strong labor demand", "fiscal expansion",
        "overseas monetary divergence", "banking sector stress", "disinflation in goods",
        "anchored inflation expectations", "softening consumer spending",
    ],
    "outcome": [
        "tighter financial conditions", "a widening output gap", "elevated term premiums",
        "compressed net interest margins", "declining real wages", "rising default rates",
    ],
    "secondary": [
        "labor market", "housing sector", "corporate credit market",
        "consumer balance sheet", "fiscal position", "external sector",
    ],
    "condition": [
        "resilient", "under pressure", "at cyclical highs", "softening",
        "broadly stable", "consistent with a soft landing",
    ],
    "policy_action": [
        "a 25-basis-point rate cut", "maintaining the current target range",
        "a 50-basis-point increase", "gradual balance sheet normalization",
        "forward guidance recalibration",
    ],
    "risk_factor": [
        "geopolitical energy price volatility", "persistent services inflation",
        "fiscal dominance", "rapid dollar appreciation", "banking sector fragility",
    ],
    "risk_outcome": [
        "reignite inflation expectations", "deepen the economic downturn",
        "destabilize sovereign debt markets", "impair monetary policy transmission",
        "widen the current account deficit",
    ],
}


def generate_mock_fomc_document(doc_id: int) -> dict[str, Any]:
    """
    Generate a single mock FOMC document with a random but coherent paragraph.

    In production this would be replaced with a real document loader
    (S3, Bloomberg, EDGAR, etc.).

    Parameters
    ----------
    doc_id : int
        Unique document identifier used for seeding and tracing.

    Returns
    -------
    dict with keys: ``doc_id``, ``text``, ``source``, ``date``.
    """
    rng = random.Random(doc_id)
    template = rng.choice(_FOMC_TEMPLATES)
    filled = template.format(
        **{k: rng.choice(v) for k, v in _TEMPLATE_VARS.items()}
    )
    year = 2015 + (doc_id % 10)
    month = 1 + (doc_id % 12)
    return {
        "doc_id": doc_id,
        "text": filled,
        "source": f"FOMC_minutes_{year}_{month:02d}",
        "date": f"{year}-{month:02d}-01",
    }


def generate_document_corpus(n: int) -> list[dict[str, Any]]:
    """Generate a corpus of ``n`` mock FOMC documents."""
    return [generate_mock_fomc_document(i) for i in range(n)]


# ---------------------------------------------------------------------------
# Ray Actors — stateful coordinators
# ---------------------------------------------------------------------------


@ray.remote
class RateLimiterActor:
    """
    Cluster-wide semaphore controlling concurrent Anthropic API calls.

    Deployed as a singleton named actor ('rate_limiter') so every worker
    on every node shares the same counter.  This prevents the cluster from
    exceeding Anthropic's per-minute request limits regardless of how many
    nodes are added.

    Implementation: a simple integer counter with acquire/release semantics,
    sufficient for our throughput target.  For stricter rate limiting
    (requests-per-second), replace with a token-bucket implementation.
    """

    def __init__(self, max_concurrent: int) -> None:
        self._max = max_concurrent
        self._in_flight: int = 0
        self._total_acquired: int = 0
        self._total_released: int = 0

    def acquire(self) -> bool:
        """
        Attempt to acquire a slot.

        Returns True if a slot was available and has been reserved,
        False if the semaphore is at capacity.  Callers should poll
        (with a short sleep) until True is returned.
        """
        if self._in_flight < self._max:
            self._in_flight += 1
            self._total_acquired += 1
            return True
        return False

    def release(self) -> None:
        """Release a previously acquired slot."""
        if self._in_flight > 0:
            self._in_flight -= 1
            self._total_released += 1

    def stats(self) -> dict[str, int]:
        return {
            "in_flight": self._in_flight,
            "total_acquired": self._total_acquired,
            "total_released": self._total_released,
        }


@ray.remote
class ResultStoreActor:
    """
    Accumulates successfully processed documents and emits progress statistics.

    In production, replace the in-memory list with writes to S3, a vector
    store, or a time-series database.  The Actor pattern ensures thread-safe
    appends across all worker nodes without explicit locking.
    """

    def __init__(self) -> None:
        self._results: list[dict[str, Any]] = []
        self._t_start: float = time.time()

    def add(self, result: dict[str, Any]) -> None:
        self._results.append(result)

    def count(self) -> int:
        return len(self._results)

    def throughput(self) -> float:
        elapsed = time.time() - self._t_start
        return len(self._results) / elapsed if elapsed > 0 else 0.0

    def progress_report(self, total: int) -> str:
        n = len(self._results)
        pct = 100 * n / total if total else 0
        tps = self.throughput()
        eta_s = (total - n) / tps if tps > 0 else float("inf")
        return (
            f"Processed {n:,}/{total:,} ({pct:.1f}%) | "
            f"Throughput: {tps:.2f} docs/s | "
            f"ETA: {eta_s/60:.1f} min"
        )

    def all_results(self) -> list[dict[str, Any]]:
        return self._results


@ray.remote
class DeadLetterActor:
    """
    Collects documents that failed all retry attempts.

    In production these would be written to an SQS dead-letter queue or
    an S3 error prefix for manual review and reprocessing.
    """

    def __init__(self) -> None:
        self._failures: list[dict[str, Any]] = []

    def add(self, doc_id: int, error: str) -> None:
        self._failures.append({"doc_id": doc_id, "error": error})
        logger.warning("Dead-lettered doc_id=%d: %s", doc_id, error)

    def count(self) -> int:
        return len(self._failures)

    def all_failures(self) -> list[dict[str, Any]]:
        return self._failures


# ---------------------------------------------------------------------------
# Ray Remote Functions — stateless workers
# ---------------------------------------------------------------------------


@ray.remote(
    num_cpus=0.5,           # extraction is I/O-bound; don't hog a full CPU
    memory=256 * 1024**2,   # 256 MB upper bound per task (Anthropic response + DAG)
    max_retries=0,          # retries handled manually with backoff below
)
def extract_dag_remote(
    doc: dict[str, Any],
    rate_limiter: ray.actor.ActorHandle,
    api_key: str,
) -> dict[str, Any]:
    """
    Extract a CausalDAG from a single document.

    Waits until the RateLimiterActor grants a slot, then calls the
    Anthropic API.  Implements exponential backoff for transient errors.
    The slot is always released in a finally block to prevent semaphore leaks.

    Parameters
    ----------
    doc : dict
        Document dict with keys ``doc_id``, ``text``, ``source``, ``date``.
    rate_limiter : ActorHandle
        Handle to the shared RateLimiterActor.
    api_key : str
        Anthropic API key (passed explicitly; env vars may not propagate to workers).

    Returns
    -------
    dict
        Original doc fields plus ``dag_json`` (serialised CausalDAG).

    Raises
    ------
    RuntimeError
        If all retry attempts are exhausted.
    """
    # Poll the rate limiter until a slot opens (non-blocking busy-wait)
    while not ray.get(rate_limiter.acquire.remote()):
        time.sleep(0.05)  # 50 ms poll interval

    try:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                dag: CausalDAG = extract_causal_dag(
                    doc["text"],
                    model="claude-sonnet-4-5",
                    api_key=api_key,
                )
                return {
                    **doc,
                    "dag_json": dag.model_dump(),
                    "node_count": len(dag.nodes),
                    "edge_count": len(dag.edges),
                }
            except Exception as exc:
                if attempt == MAX_RETRIES:
                    raise RuntimeError(
                        f"doc_id={doc['doc_id']} failed after {MAX_RETRIES} attempts: {exc}"
                    ) from exc
                delay = RETRY_BASE_DELAY_S * (2 ** (attempt - 1))
                logger.warning(
                    "doc_id=%d attempt %d/%d failed (%s). Retrying in %.1fs …",
                    doc["doc_id"], attempt, MAX_RETRIES, exc, delay,
                )
                time.sleep(delay)
    finally:
        ray.get(rate_limiter.release.remote())


@ray.remote(num_cpus=0.25)  # sequencification is CPU-bound but trivially fast
def sequencify_remote(extracted: dict[str, Any]) -> dict[str, Any]:
    """
    Convert an extracted DAG dict into a text sequence.

    Accepts the output of extract_dag_remote, reconstructs the CausalDAG
    Pydantic object, runs sequencify_dag(), and returns the enriched record.

    Parameters
    ----------
    extracted : dict
        Output of extract_dag_remote: doc fields + ``dag_json``.

    Returns
    -------
    dict
        All input fields plus ``sequence`` (the structured text string).
    """
    dag = CausalDAG.model_validate(extracted["dag_json"])
    sequence = sequencify_dag(dag)
    return {**extracted, "sequence": sequence}


@ray.remote(num_cpus=0.1)
def persist_result_remote(
    sequencified: dict[str, Any],
    output_dir: str,
    result_store: ray.actor.ActorHandle,
) -> int:
    """
    Write a sequencified record to disk (mock persistence layer).

    In production, replace the local write with:
    - S3: boto3.client('s3').put_object(...)
    - Vector store: qdrant_client.upsert(...)
    - Feature store: feast.FeatureStore.write_to_online_store(...)

    Parameters
    ----------
    sequencified : dict
        Output of sequencify_remote.
    output_dir : str
        Local path prefix for output files.
    result_store : ActorHandle
        Handle to the shared ResultStoreActor.

    Returns
    -------
    int
        doc_id of the persisted document.
    """
    doc_id: int = sequencified["doc_id"]
    out_path = Path(output_dir) / f"doc_{doc_id:07d}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write the sequence + metadata (exclude heavy dag_json to save space)
    payload = {
        "doc_id": doc_id,
        "source": sequencified.get("source"),
        "date": sequencified.get("date"),
        "node_count": sequencified.get("node_count"),
        "edge_count": sequencified.get("edge_count"),
        "sequence": sequencified["sequence"],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    ray.get(result_store.add.remote({"doc_id": doc_id, "path": str(out_path)}))
    return doc_id


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Tunable knobs for the distributed pipeline."""

    n_documents: int = 100_000
    submit_batch_size: int = SUBMIT_BATCH_SIZE
    max_concurrent_api_calls: int = MAX_CONCURRENT_API_CALLS
    output_dir: str = str(OUTPUT_DIR)
    progress_every_n: int = 500      # log a progress report every N completions
    api_key: str = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", "mock-key")
    )


def run_pipeline(config: PipelineConfig) -> dict[str, Any]:
    """
    Orchestrate the full three-stage pipeline across the Ray cluster.

    Architecture
    ~~~~~~~~~~~~
    1. Singleton actors (RateLimiter, ResultStore, DeadLetter) are created
       with ``get_if_exists=True`` so re-runs on the same cluster are idempotent.
    2. Documents are submitted in ``submit_batch_size`` chunks to bound object-store
       memory usage.  A chunk is submitted, then ray.wait() drains completed futures
       before the next chunk is submitted — a classic producer-consumer pattern.
    3. Each document traverses three remote tasks in a linear DAG:
           extract_dag_remote  →  sequencify_remote  →  persist_result_remote
       Because each stage returns an ObjectRef, Ray automatically pipelines
       them: sequencify starts as soon as extract finishes, without the
       orchestrator ever touching the intermediate value.
    4. Failures are routed to the DeadLetterActor; the pipeline never raises
       and always completes even under partial failures.

    Parameters
    ----------
    config : PipelineConfig

    Returns
    -------
    dict
        Summary statistics: docs processed, docs failed, total wall time,
        throughput (docs/s).
    """
    logger.info(
        "Initialising pipeline | n_docs=%d | max_api_concurrent=%d",
        config.n_documents,
        config.max_concurrent_api_calls,
    )

    # ── Singleton actors ─────────────────────────────────────────────────────
    rate_limiter = RateLimiterActor.options(
        name="rate_limiter",
        get_if_exists=True,
        lifetime="detached",
    ).remote(config.max_concurrent_api_calls)

    result_store = ResultStoreActor.options(
        name="result_store",
        get_if_exists=True,
        lifetime="detached",
    ).remote()

    dead_letter = DeadLetterActor.options(
        name="dead_letter",
        get_if_exists=True,
        lifetime="detached",
    ).remote()

    # ── Generate document corpus ─────────────────────────────────────────────
    logger.info("Generating %d mock FOMC documents …", config.n_documents)
    corpus = generate_document_corpus(config.n_documents)

    # ── Submit pipeline in sliding batches ───────────────────────────────────
    t_pipeline_start = time.perf_counter()
    pending_persist_refs: list[ray.ObjectRef] = []
    completed_count = 0

    for chunk_start in range(0, len(corpus), config.submit_batch_size):
        chunk = corpus[chunk_start : chunk_start + config.submit_batch_size]

        for doc in chunk:
            # Stage 1: extract (returns ObjectRef[dict])
            extract_ref = extract_dag_remote.remote(doc, rate_limiter, config.api_key)

            # Stage 2: sequencify — chained directly on the ObjectRef.
            # Ray will not schedule this until extract_ref resolves,
            # but the orchestrator does NOT block here.
            seq_ref = sequencify_remote.remote(extract_ref)

            # Stage 3: persist — chained on seq_ref
            persist_ref = persist_result_remote.remote(
                seq_ref,
                config.output_dir,
                result_store,
            )
            pending_persist_refs.append(persist_ref)

        # Drain completed futures from this chunk before submitting the next.
        # ray.wait() returns as soon as `num_returns` futures are ready,
        # so we stream results rather than blocking on all.
        while pending_persist_refs:
            done_refs, pending_persist_refs = ray.wait(
                pending_persist_refs,
                num_returns=min(len(chunk), len(pending_persist_refs)),
                timeout=60.0,  # hard timeout per drain cycle
            )

            for ref in done_refs:
                try:
                    doc_id: int = ray.get(ref)
                    completed_count += 1
                    if completed_count % config.progress_every_n == 0:
                        report = ray.get(result_store.progress_report.remote(config.n_documents))
                        logger.info(report)
                except Exception as exc:
                    # Extract doc_id from the failed ref's lineage if possible
                    ray.get(dead_letter.add.remote(-1, str(exc)))

    # Final drain for any stragglers
    if pending_persist_refs:
        try:
            ray.get(pending_persist_refs, timeout=120.0)
        except Exception as exc:
            logger.warning("Straggler drain error: %s", exc)

    # ── Collect final statistics ─────────────────────────────────────────────
    wall_time_s = time.perf_counter() - t_pipeline_start
    n_success = ray.get(result_store.count.remote())
    n_failed = ray.get(dead_letter.count.remote())
    rl_stats = ray.get(rate_limiter.stats.remote())

    summary = {
        "n_documents": config.n_documents,
        "n_success": n_success,
        "n_failed": n_failed,
        "wall_time_s": round(wall_time_s, 2),
        "throughput_docs_per_s": round(n_success / wall_time_s, 2) if wall_time_s > 0 else 0,
        "rate_limiter_stats": rl_stats,
    }

    logger.info(
        "Pipeline complete | success=%d | failed=%d | wall=%.1fs | tps=%.2f",
        n_success,
        n_failed,
        wall_time_s,
        summary["throughput_docs_per_s"],
    )
    return summary


# ---------------------------------------------------------------------------
# Local simulation mode (no real cluster or API needed)
# ---------------------------------------------------------------------------


def run_local_simulation(n_docs: int) -> None:
    """
    Demonstrate the pipeline architecture locally without real API calls.

    Replaces the Anthropic extraction step with mock CausalDAG objects so
    the full Ray task graph, actor coordination, and sequencification logic
    can be exercised on a laptop.

    This is the mode executed when you run ``python distributed_batch.py``.
    """
    logger.info("=== LOCAL SIMULATION MODE (no real API calls) ===")
    logger.info("Simulating pipeline for %d documents …", n_docs)

    ray.init(ignore_reinit_error=True, num_cpus=4, num_gpus=0)
    logger.info("Ray cluster: %s", ray.cluster_resources())

    @ray.remote(num_cpus=0.5)
    def mock_extract_remote(doc: dict[str, Any]) -> dict[str, Any]:
        """
        Simulate extraction by loading a real DAG from data/ if available,
        otherwise constructing a minimal synthetic CausalDAG.
        """
        import json
        from pathlib import Path
        from extractor import CausalDAG, CausalNode, CausalEdge

        data_files = sorted(Path("data").glob("*.json"))
        if data_files:
            path = data_files[doc["doc_id"] % len(data_files)]
            dag = CausalDAG.model_validate(json.loads(path.read_text()))
        else:
            # Synthetic minimal DAG
            dag = CausalDAG(
                source_text_snippet=doc["text"][:200],
                nodes=[
                    CausalNode(id="policy_rate", label="Policy Rate",
                               description="Central bank rate", variable_type="policy"),
                    CausalNode(id="inflation", label="Inflation",
                               description="CPI inflation", variable_type="price"),
                ],
                edges=[
                    CausalEdge(cause="policy_rate", effect="inflation",
                               confounder=None,
                               mechanism="Higher rates compress demand and reduce price pressures.",
                               confidence_score=0.9, edge_sign="negative"),
                ],
                summary="Policy rate negatively causes inflation via demand compression.",
            )
        time.sleep(random.uniform(0.01, 0.05))  # simulate network latency
        return {**doc, "dag_json": dag.model_dump(),
                "node_count": len(dag.nodes), "edge_count": len(dag.edges)}

    corpus = generate_document_corpus(n_docs)
    t_start = time.perf_counter()

    # Submit all docs; chain sequencify onto each extract future
    futures = []
    for doc in corpus:
        extract_ref = mock_extract_remote.remote(doc)
        seq_ref = sequencify_remote.remote(extract_ref)
        futures.append(seq_ref)

    # Stream results as they complete
    completed, failed = 0, 0
    remaining = list(futures)
    while remaining:
        done, remaining = ray.wait(remaining, num_returns=min(10, len(remaining)), timeout=30.0)
        for ref in done:
            try:
                result = ray.get(ref)
                completed += 1
                if completed % max(1, n_docs // 10) == 0:
                    elapsed = time.perf_counter() - t_start
                    logger.info(
                        "Progress: %d/%d (%.0f%%) | %.2f docs/s",
                        completed, n_docs, 100 * completed / n_docs,
                        completed / elapsed,
                    )
            except Exception as exc:
                failed += 1
                logger.warning("Task failed: %s", exc)

    wall = time.perf_counter() - t_start
    logger.info(
        "Simulation complete | success=%d | failed=%d | wall=%.2fs | tps=%.2f",
        completed, failed, wall, completed / wall if wall > 0 else 0,
    )
    ray.shutdown()


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed FOMC batch pipeline")
    parser.add_argument(
        "--docs",
        type=int,
        default=100,
        help="Number of documents to simulate (default: 100). "
             "Use --docs 100000 with a real cluster and API key.",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        default=False,
        help="Run the real pipeline (requires ANTHROPIC_API_KEY and a Ray cluster).",
    )
    args = parser.parse_args()

    if args.real:
        ray.init(address=os.environ.get("RAY_ADDRESS", "auto"))
        config = PipelineConfig(n_documents=args.docs)
        summary = run_pipeline(config)
        print(json.dumps(summary, indent=2))
        ray.shutdown()
    else:
        run_local_simulation(n_docs=args.docs)
