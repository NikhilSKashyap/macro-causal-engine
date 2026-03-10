"""
sequencifier.py
---------------
ML data pipeline: Pydantic CausalDAG → topologically-sorted text sequences
→ tokenized PyTorch batches ready for autoregressive causal language modelling.

Pipeline overview
~~~~~~~~~~~~~~~~~
1. build_dag_graph()      – constructs an nx.DiGraph from a CausalDAG.
2. topo_sort_edges()      – returns edges ordered by a topological walk of nodes.
3. sequencify_dag()       – serialises the sorted edges into a structured text
                            sequence using a domain-specific micro-grammar.
4. load_tokenizer()       – attempts Llama-3 → GPT-2 → DummyTokenizer in order.
5. CausalDAGDataset       – PyTorch Dataset: DAG list → tokenized sequences.
6. build_dataloader()     – wraps the Dataset in a DataLoader with dynamic padding.

Sequence grammar (one edge per line, prefixed by a DAG header):
    <|dag_start|>
    [CTX] <source_text_snippet>
    [NODE:<type>] <label>          (one per node, in topo order)
    [<sign>|<conf>] <cause_label> -> <effect_label> | Mechanism: <text> | Confounder: <label|NONE>
    <|dag_end|>

Sign tokens:  [+] positive  [-] negative  [~] ambiguous
Conf bucket:  HIGH ≥ 0.8   MED ≥ 0.5   LOW < 0.5
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import networkx as nx
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from extractor import CausalDAG, CausalEdge, CausalNode

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SIGN_TOKEN: dict[str, str] = {
    "positive": "[+]",
    "negative": "[-]",
    "ambiguous": "[~]",
}

CONF_BUCKET: dict[str, float] = {
    "HIGH": 0.8,
    "MED": 0.5,
}

SEQ_START = "<|dag_start|>"
SEQ_END = "<|dag_end|>"
PAD_TOKEN = "<|pad|>"

# ---------------------------------------------------------------------------
# 1. Graph construction
# ---------------------------------------------------------------------------


def build_dag_graph(dag: CausalDAG) -> nx.DiGraph:
    """
    Build a NetworkX DiGraph from a CausalDAG Pydantic object.

    Nodes carry the full CausalNode as a ``data`` attribute so downstream
    functions can access labels and variable types without re-querying.

    Parameters
    ----------
    dag : CausalDAG
        A validated Pydantic DAG produced by extractor.py.

    Returns
    -------
    nx.DiGraph
        Directed graph; nodes keyed by CausalNode.id, edges keyed by
        (cause_id, effect_id) with the CausalEdge stored as ``data``.

    Raises
    ------
    nx.NetworkXUnfeasible
        If the graph contains a cycle (violates the DAG constraint).
    """
    g: nx.DiGraph = nx.DiGraph()

    node_map: dict[str, CausalNode] = {n.id: n for n in dag.nodes}

    for node in dag.nodes:
        g.add_node(node.id, data=node)

    for edge in dag.edges:
        g.add_edge(edge.cause, edge.effect, data=edge)

    if not nx.is_directed_acyclic_graph(g):
        cycles = list(nx.simple_cycles(g))
        raise nx.NetworkXUnfeasible(
            f"CausalDAG contains {len(cycles)} cycle(s): {cycles}. "
            "A DAG must be acyclic."
        )

    return g


# ---------------------------------------------------------------------------
# 2. Topological sort → ordered edge list
# ---------------------------------------------------------------------------


def topo_sort_edges(dag: CausalDAG, g: nx.DiGraph) -> list[CausalEdge]:
    """
    Return dag.edges ordered by a topological traversal of the graph.

    Algorithm
    ~~~~~~~~~
    - Compute a topological ordering of nodes via Kahn's algorithm
      (nx.topological_sort uses DFS; both are correct).
    - For each node in that order, emit all outgoing edges in the order
      they appear in dag.edges (preserving extraction-time ordering within
      the same source node).

    This produces a sequence where causes always appear before their effects,
    which is exactly the left-to-right causal reading order required for
    autoregressive language modelling.

    Parameters
    ----------
    dag : CausalDAG
        Original Pydantic DAG (used to preserve per-node edge ordering).
    g : nx.DiGraph
        Graph built from the same DAG via build_dag_graph().

    Returns
    -------
    list[CausalEdge]
        Edges in topological cause-first order. Edges whose cause node
        appears earlier in the topological sort come first.
    """
    topo_order: list[str] = list(nx.topological_sort(g))
    rank: dict[str, int] = {node_id: i for i, node_id in enumerate(topo_order)}

    # Build a lookup: cause_id → [edges] preserving extraction order
    cause_to_edges: dict[str, list[CausalEdge]] = {}
    for edge in dag.edges:
        cause_to_edges.setdefault(edge.cause, []).append(edge)

    sorted_edges: list[CausalEdge] = []
    for node_id in topo_order:
        sorted_edges.extend(cause_to_edges.get(node_id, []))

    return sorted_edges


# ---------------------------------------------------------------------------
# 3. Sequencifier — DAG → text
# ---------------------------------------------------------------------------


def _conf_label(score: float) -> str:
    """Map a float confidence score to a bucketed string label."""
    if score >= CONF_BUCKET["HIGH"]:
        return "HIGH"
    if score >= CONF_BUCKET["MED"]:
        return "MED"
    return "LOW"


def sequencify_dag(dag: CausalDAG) -> str:
    """
    Serialise a CausalDAG into a structured text sequence for LLM ingestion.

    Format
    ~~~~~~
    ::

        <|dag_start|>
        [CTX] <source_text_snippet>
        [NODE:policy] Federal Funds Rate
        [NODE:price] Headline PCE Inflation
        ...
        [+|HIGH] Monetary Tightening Cycle -> Aggregate Demand | Mechanism: ... | Confounder: NONE
        [-|HIGH] Aggregate Demand -> Unemployment Rate | Mechanism: ... | Confounder: NONE
        [+|MED]  Premature Policy Easing -> Inflation Expectations | ... | Confounder: Geopolitical Energy Price Volatility
        <|dag_end|>

    Design rationale
    ~~~~~~~~~~~~~~~~
    - **Topological ordering** ensures the autoregressive model always sees a
      cause token before its effect token, matching the natural left-to-right
      reading direction of causal statements.
    - **[NODE:<type>]** lines prime the model with the full variable inventory
      before any edges appear, reducing hallucination of unknown variables.
    - **[<sign>|<conf_bucket>]** edge prefixes encode two dense signals in a
      single token-efficient tag, avoiding verbose natural-language repetition.
    - **Confounder: NONE** is explicit rather than absent so the model learns
      to actively predict confounders rather than treating their absence as
      uninformative padding.

    Parameters
    ----------
    dag : CausalDAG
        Validated Pydantic DAG.

    Returns
    -------
    str
        A single multi-line string representing the full DAG as a text sequence.
    """
    g = build_dag_graph(dag)
    sorted_edges = topo_sort_edges(dag, g)

    # Node label lookup
    node_label: dict[str, str] = {n.id: n.label for n in dag.nodes}
    node_type: dict[str, str] = {n.id: n.variable_type for n in dag.nodes}

    # Topo-ordered node list (for the NODE header block)
    topo_node_ids: list[str] = list(nx.topological_sort(g))

    lines: list[str] = [SEQ_START]

    # Context header
    lines.append(f"[CTX] {dag.source_text_snippet.strip()}")

    # Node inventory block
    for node_id in topo_node_ids:
        label = node_label.get(node_id, node_id)
        vtype = node_type.get(node_id, "unknown")
        lines.append(f"[NODE:{vtype}] {label}")

    # Edge sequence in topological order
    for edge in sorted_edges:
        cause_lbl = node_label.get(edge.cause, edge.cause)
        effect_lbl = node_label.get(edge.effect, edge.effect)
        sign_tok = SIGN_TOKEN.get(edge.edge_sign, "[~]")
        conf_tok = _conf_label(edge.confidence_score)
        conf_val = f"{edge.confidence_score:.2f}"

        confounder_str = (
            node_label.get(edge.confounder, edge.confounder)
            if edge.confounder
            else "NONE"
        )

        lines.append(
            f"[{sign_tok}|{conf_tok}({conf_val})] "
            f"{cause_lbl} -> {effect_lbl} "
            f"| Mechanism: {edge.mechanism} "
            f"| Confounder: {confounder_str}"
        )

    lines.append(SEQ_END)
    return "\n".join(lines)


def sequencify_batch(dags: list[CausalDAG]) -> list[str]:
    """
    Sequencify a list of CausalDAGs into a list of text sequences.

    Parameters
    ----------
    dags : list[CausalDAG]
        DAGs to serialise.

    Returns
    -------
    list[str]
        One text sequence per DAG, in input order.
    """
    return [sequencify_dag(dag) for dag in dags]


# ---------------------------------------------------------------------------
# 4. Tokenizer loading (Llama-3 → GPT-2 → DummyTokenizer)
# ---------------------------------------------------------------------------


class DummyTokenizer:
    """
    Minimal word-level tokenizer used when no Hugging Face model is available.

    Builds a vocabulary from the sequences it sees on first use, assigns
    integer ids, and pads/truncates to a fixed max_length.  Not suitable for
    production training — exists solely to keep the pipeline runnable offline.

    Attributes
    ----------
    vocab : dict[str, int]
        Word → integer id mapping (built lazily on first __call__).
    max_length : int
        Truncation / padding target length.
    pad_token_id : int
        Token id used for padding (always 0).
    """

    PAD_ID = 0
    UNK_ID = 1
    BOS_ID = 2
    EOS_ID = 3
    _RESERVED = 4  # first assignable id

    def __init__(self, max_length: int = 512) -> None:
        self.max_length = max_length
        self.pad_token_id = self.PAD_ID
        self.bos_token_id = self.BOS_ID
        self.eos_token_id = self.EOS_ID
        self.vocab: dict[str, int] = {
            PAD_TOKEN: self.PAD_ID,
            "<|unk|>": self.UNK_ID,
            "<|bos|>": self.BOS_ID,
            "<|eos|>": self.EOS_ID,
        }
        self._next_id: int = self._RESERVED
        self.model_max_length: int = max_length

    def _encode_word(self, word: str) -> int:
        if word not in self.vocab:
            self.vocab[word] = self._next_id
            self._next_id += 1
        return self.vocab[word]

    def __call__(
        self,
        texts: list[str] | str,
        max_length: int | None = None,
        padding: str = "max_length",
        truncation: bool = True,
        return_tensors: str | None = "pt",
    ) -> dict[str, Tensor]:
        """
        Tokenize one or more text strings.

        Returns a dict with ``input_ids`` and ``attention_mask`` tensors,
        matching the interface expected by CausalDAGDataset.
        """
        if isinstance(texts, str):
            texts = [texts]
        target_len = max_length or self.max_length

        all_ids: list[list[int]] = []
        all_masks: list[list[int]] = []

        for text in texts:
            tokens = text.split()
            ids = [self._encode_word(t) for t in tokens]

            if truncation:
                ids = ids[:target_len]

            mask = [1] * len(ids)

            if padding == "max_length":
                pad_len = target_len - len(ids)
                ids = ids + [self.PAD_ID] * pad_len
                mask = mask + [0] * pad_len

            all_ids.append(ids)
            all_masks.append(mask)

        result: dict[str, Any] = {
            "input_ids": torch.tensor(all_ids, dtype=torch.long),
            "attention_mask": torch.tensor(all_masks, dtype=torch.long),
        }
        return result


def load_tokenizer(
    preferred: str = "meta-llama/Meta-Llama-3-8B",
    fallback: str = "gpt2",
    max_length: int = 512,
) -> Any:
    """
    Load a tokenizer with a three-tier fallback strategy.

    Tiers
    ~~~~~
    1. ``preferred`` model (default: Llama-3-8B) — requires HF auth token.
    2. ``fallback`` model (default: GPT-2) — publicly available, no auth.
    3. DummyTokenizer — offline word-level fallback.

    Both HF tokenizers are configured with:
    - ``padding_side = "right"`` (required for decoder-only CLM)
    - ``pad_token`` set to ``eos_token`` if absent (Llama-3 has no pad token)

    Parameters
    ----------
    preferred : str
        HuggingFace model id to try first.
    fallback : str
        HuggingFace model id to try if preferred fails.
    max_length : int
        Sequence length cap; also sets tokenizer.model_max_length.

    Returns
    -------
    A tokenizer instance with a callable ``__call__(texts, ...)`` interface.
    """
    try:
        from transformers import AutoTokenizer

        for model_id in (preferred, fallback):
            try:
                tok = AutoTokenizer.from_pretrained(model_id)
                tok.model_max_length = max_length
                tok.padding_side = "right"
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token
                    tok.pad_token_id = tok.eos_token_id
                logger.info("Loaded tokenizer: %s", model_id)
                return tok
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not load '%s': %s. Trying next.", model_id, exc)

    except ImportError:
        logger.warning("`transformers` not installed. Falling back to DummyTokenizer.")

    logger.warning("Using DummyTokenizer (offline word-level fallback).")
    return DummyTokenizer(max_length=max_length)


# ---------------------------------------------------------------------------
# 5. PyTorch Dataset
# ---------------------------------------------------------------------------


class CausalDAGDataset(Dataset):
    """
    PyTorch Dataset that converts a list of CausalDAG objects into tokenized
    tensors ready for autoregressive causal language modelling.

    Each item is a dict with three keys:

    ``input_ids`` : LongTensor [seq_len]
        Token ids of the full sequencified DAG.
    ``attention_mask`` : LongTensor [seq_len]
        1 for real tokens, 0 for padding.
    ``labels`` : LongTensor [seq_len]
        Identical to ``input_ids`` — the standard convention for CLM where
        the training loop (or HuggingFace Trainer) applies the causal shift
        internally. Padding positions are set to -100 so CrossEntropyLoss
        ignores them.

    Parameters
    ----------
    dags : list[CausalDAG]
        Validated Pydantic DAG objects (e.g. loaded from data/*.json).
    tokenizer : any
        A callable tokenizer with the HuggingFace interface.
    max_length : int
        Maximum sequence length; sequences are truncated to this value.
    """

    def __init__(
        self,
        dags: list[CausalDAG],
        tokenizer: Any,
        max_length: int = 512,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Pre-sequencify all DAGs at construction time so __getitem__ is O(1)
        self.sequences: list[str] = sequencify_batch(dags)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """
        Tokenize and return a single DAG sequence.

        Returns
        -------
        dict with keys ``input_ids``, ``attention_mask``, ``labels``.
        """
        encoded: dict[str, Tensor] = self.tokenizer(
            self.sequences[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids: Tensor = encoded["input_ids"].squeeze(0)       # [seq_len]
        attention_mask: Tensor = encoded["attention_mask"].squeeze(0)  # [seq_len]

        # Labels: copy of input_ids with padding positions masked as -100
        labels: Tensor = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ---------------------------------------------------------------------------
# 6. DataLoader factory
# ---------------------------------------------------------------------------


def _collate_fn(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """
    Collate a list of dataset items into a batch with dynamic right-padding.

    Even though CausalDAGDataset pads to max_length at item level, this
    collate function re-pads to the longest sequence *in the batch*, reducing
    wasted compute when batch items are uniformly shorter than max_length.

    Parameters
    ----------
    batch : list[dict[str, Tensor]]
        List of items returned by CausalDAGDataset.__getitem__.

    Returns
    -------
    dict[str, Tensor]
        ``input_ids``      [batch, max_batch_len]
        ``attention_mask`` [batch, max_batch_len]
        ``labels``         [batch, max_batch_len]
    """
    # Find the true length of each item (up to last non-pad token)
    real_lengths: list[int] = [
        int((item["attention_mask"] == 1).sum().item()) for item in batch
    ]
    max_len: int = max(real_lengths)

    input_ids_batch: list[Tensor] = []
    attention_mask_batch: list[Tensor] = []
    labels_batch: list[Tensor] = []

    for item, real_len in zip(batch, real_lengths):
        input_ids_batch.append(item["input_ids"][:max_len])
        attention_mask_batch.append(item["attention_mask"][:max_len])
        labels_batch.append(item["labels"][:max_len])

    return {
        "input_ids": torch.stack(input_ids_batch),           # [B, L]
        "attention_mask": torch.stack(attention_mask_batch),  # [B, L]
        "labels": torch.stack(labels_batch),                  # [B, L]
    }


def build_dataloader(
    dags: list[CausalDAG],
    tokenizer: Any | None = None,
    max_length: int = 512,
    batch_size: int = 2,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Build a PyTorch DataLoader from a list of CausalDAG objects.

    Handles tokenizer initialisation if none is supplied.

    Parameters
    ----------
    dags : list[CausalDAG]
        Validated Pydantic DAG objects.
    tokenizer : any | None
        Pre-loaded tokenizer. If None, load_tokenizer() is called with defaults.
    max_length : int
        Truncation / padding target length per sequence.
    batch_size : int
        Number of DAGs per batch.
    shuffle : bool
        Whether to shuffle the dataset each epoch.
    num_workers : int
        Subprocess workers for data loading (0 = main process only).

    Returns
    -------
    DataLoader
        Ready-to-iterate DataLoader yielding dicts of
        ``{input_ids, attention_mask, labels}`` tensors.
    """
    if tokenizer is None:
        tokenizer = load_tokenizer(max_length=max_length)

    dataset = CausalDAGDataset(dags=dags, tokenizer=tokenizer, max_length=max_length)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )


# ---------------------------------------------------------------------------
# 7. Utilities — load DAGs from disk
# ---------------------------------------------------------------------------


def load_dags_from_dir(data_dir: str | Path) -> list[CausalDAG]:
    """
    Load all ``*.json`` files from a directory as CausalDAG objects.

    Parameters
    ----------
    data_dir : str | Path
        Directory containing JSON files produced by extractor.py.

    Returns
    -------
    list[CausalDAG]
        Validated CausalDAG objects, sorted by filename for reproducibility.
    """
    data_path = Path(data_dir)
    json_files = sorted(data_path.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {data_path.resolve()}")

    dags: list[CausalDAG] = []
    for path in json_files:
        raw = json.loads(path.read_text(encoding="utf-8"))
        dags.append(CausalDAG.model_validate(raw))
        logger.info("Loaded DAG: %s", path.name)

    return dags


# ---------------------------------------------------------------------------
# 8. CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import textwrap

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    # ── Load DAGs from data/ directory ──────────────────────────────────────
    dags = load_dags_from_dir("data")
    print(f"\nLoaded {len(dags)} DAGs.\n")

    # ── Show the sequencified text for DAG 0 ────────────────────────────────
    seq = sequencify_dag(dags[0])
    print("=" * 72)
    print("SEQUENCIFIED DAG [0] — FOMC Rate Cut")
    print("=" * 72)
    print(seq)
    print()

    # ── Build DataLoader ─────────────────────────────────────────────────────
    tokenizer = load_tokenizer(max_length=512)
    loader = build_dataloader(
        dags=dags,
        tokenizer=tokenizer,
        max_length=512,
        batch_size=2,
        shuffle=False,
    )

    print("=" * 72)
    print(f"DataLoader: {len(loader.dataset)} samples | batch_size=2")
    print("=" * 72)

    for batch_idx, batch in enumerate(loader):
        print(
            f"\nBatch {batch_idx + 1}:"
            f"\n  input_ids      shape : {batch['input_ids'].shape}"
            f"\n  attention_mask shape : {batch['attention_mask'].shape}"
            f"\n  labels         shape : {batch['labels'].shape}"
            f"\n  Non-pad tokens (mask==1): "
            f"{batch['attention_mask'].sum(dim=1).tolist()}"
            f"\n  Label -100 count (pad masked): "
            f"{(batch['labels'] == -100).sum(dim=1).tolist()}"
        )
