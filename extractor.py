"""
extractor.py
------------
Macro-economic text ingestion and causal DAG extraction pipeline.

Uses Claude 3.5 Sonnet via the Anthropic tool-calling (structured outputs) API to
parse dense macroeconomic prose and return a typed Pydantic Causal DAG object.

Architecture overview
~~~~~~~~~~~~~~~~~~~~~
1. MOCK_PARAGRAPHS   – five hand-crafted, highly complex macro paragraphs that
                       mirror real-world FOMC minutes, BIS working papers, and
                       IMF surveillance reports.
2. Pydantic schema   – CausalNode, CausalEdge, CausalDAG.
3. extract_causal_dag() – calls Claude with tool_use, validates the returned JSON
                         against the Pydantic schema, and returns a CausalDAG.
4. batch_extract()   – convenience wrapper to process a list of paragraphs.
"""

from __future__ import annotations

import json
import os
from typing import Any

import anthropic
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# 1. Mock dataset
# ---------------------------------------------------------------------------

MOCK_PARAGRAPHS: list[str] = [
    # ── Paragraph 1 ── FOMC rate-cut deliberation
    (
        "Members noted that, while headline PCE inflation had decelerated materially to 2.4 percent "
        "on a twelve-month basis, the stickiness of services-ex-housing components—particularly "
        "non-market services tied to wage-sensitive industries—continued to create upside risks. "
        "Several participants argued that the cumulative 525 basis-point tightening cycle had "
        "succeeded in compressing aggregate demand sufficiently to bring the unemployment rate from "
        "its cycle low of 3.4 percent to 4.1 percent, thereby reducing wage-cost pressures. "
        "However, a minority expressed concern that premature easing could reignite inflation "
        "expectations, especially given persistent geopolitical energy price volatility. The "
        "Committee ultimately judged that a 25-basis-point cut to the federal funds rate target "
        "range was appropriate, contingent on continued progress toward the 2 percent inflation "
        "objective, and signaled that the path of subsequent rate adjustments would remain data-"
        "dependent, with particular attention to labor market developments and core goods disinflation."
    ),
    # ── Paragraph 2 ── Yield curve inversion and credit dynamics
    (
        "The persistent inversion of the Treasury yield curve—with the 2-year note yielding "
        "approximately 65 basis points above the 10-year bond for the fifth consecutive quarter—"
        "continued to suppress net interest margins at regional banks, inducing a meaningful "
        "tightening of commercial and industrial lending standards. Credit growth to small and "
        "medium-sized enterprises decelerated to 1.2 percent year-over-year, its slowest pace "
        "since the 2009 trough, as loan officers simultaneously raised collateral requirements and "
        "reduced credit lines. This broad-based deleveraging transmitted contractionary pressure to "
        "capital expenditure across non-financial corporates, with private fixed investment falling "
        "0.8 percent in real terms quarter-over-quarter. Simultaneously, the sovereign spread "
        "between investment-grade and high-yield indices widened by 140 basis points, reflecting "
        "rising default risk premia amid deteriorating earnings revisions and slowing revenue growth, "
        "further tightening financial conditions for leveraged borrowers dependent on refinancing."
    ),
    # ── Paragraph 3 ── Commodity-inflation pass-through and exchange rate
    (
        "A 38-percent appreciation of West Texas Intermediate crude oil prices, catalyzed by OPEC+ "
        "supply curtailments and renewed Middle Eastern geopolitical tensions, transmitted through "
        "the energy import channel to a 1.9-percentage-point increase in producer price inflation "
        "for intermediate goods. Firms operating with limited pricing power—concentrated in "
        "consumer staples and discretionary retail—absorbed a portion of the margin compression "
        "rather than fully passing through cost increases, constraining profit growth. The "
        "concomitant depreciation of the U.S. dollar index by 4.3 percent amplified import-price "
        "inflation, particularly for economies with high dollar-invoiced trade shares, creating "
        "imported inflation feedback loops. Central banks in emerging-market economies faced a "
        "classic trilemma: tightening domestic policy rates to defend currency pegs risked "
        "deepening domestic recessions, while allowing exchange rate depreciation exacerbated "
        "balance-of-payments pressures on dollar-denominated sovereign debt obligations."
    ),
    # ── Paragraph 4 ── Labor market, productivity, and wage-price spiral risk
    (
        "Despite headline payroll growth averaging 172,000 jobs per month over the trailing "
        "six-month window, underlying labor market dynamics revealed a compositional deterioration: "
        "full-time employment contracted by 0.3 percent while part-time employment for economic "
        "reasons rose 5.7 percent, suggesting employers were adjusting margins by reducing hours "
        "rather than initiating layoffs. Real average hourly earnings grew 1.1 percent "
        "year-over-year after adjusting for CPI, compressing household purchasing power and "
        "contributing to a 0.4-percentage-point decline in the personal savings rate. Labor "
        "productivity growth—at 0.9 percent annualized—failed to keep pace with nominal wage "
        "growth of 4.3 percent, widening unit labor costs by 3.2 percent and sustaining upward "
        "pressure on services-sector prices. Economists debated whether the observed wage "
        "stickiness reflected genuine labor market tightness, anchored inflation expectations "
        "from pandemic-era shocks, or structural supply-side rigidities resulting from demographic "
        "shifts and reduced prime-age labor force participation."
    ),
    # ── Paragraph 5 ── Fiscal dominance, sovereign debt sustainability, and monetary space
    (
        "With the federal debt-to-GDP ratio approaching 122 percent and the Congressional Budget "
        "Office projecting net interest outlays to consume 3.6 percent of GDP within the decade, "
        "sovereign debt sustainability concerns began to impose binding constraints on monetary "
        "policy independence. Market participants increasingly priced the risk of fiscal dominance—"
        "the scenario in which the central bank is compelled to monetize deficit spending to "
        "prevent sovereign debt rollover crises—into long-duration Treasury term premiums, which "
        "rose 48 basis points over the quarter. Widening fiscal deficits crowded out private "
        "investment by sustaining real long-term interest rates at levels incompatible with "
        "positive NPV capital projects at historical return thresholds. Simultaneously, elevated "
        "public debt reduced the fiscal multiplier on future stimulus packages, as Ricardian "
        "equivalence effects led households to increase precautionary savings in anticipation of "
        "future tax liabilities, partially offsetting expansionary fiscal impulses and suppressing "
        "aggregate demand growth."
    ),
]

# ---------------------------------------------------------------------------
# 2. Pydantic schema – Causal DAG
# ---------------------------------------------------------------------------


class CausalNode(BaseModel):
    """
    Represents a single economic variable (vertex) in the causal DAG.

    Attributes
    ----------
    id : str
        Short machine-readable identifier, e.g. ``"federal_funds_rate"``.
    label : str
        Human-readable name, e.g. ``"Federal Funds Rate"``.
    description : str
        Brief characterisation of the variable and its role in the passage.
    variable_type : str
        Ontological category: one of ``"policy"``, ``"price"``, ``"real"``,
        ``"financial"``, ``"expectational"``, or ``"external"``.
    """

    id: str = Field(..., description="Short snake_case identifier for the node.")
    label: str = Field(..., description="Human-readable label for the economic variable.")
    description: str = Field(..., description="Brief description of the variable's role.")
    variable_type: str = Field(
        ...,
        description=(
            "Ontological category. Must be one of: "
            "'policy', 'price', 'real', 'financial', 'expectational', 'external'."
        ),
    )

    @field_validator("variable_type")
    @classmethod
    def validate_variable_type(cls, v: str) -> str:
        allowed = {"policy", "price", "real", "financial", "expectational", "external"}
        if v not in allowed:
            raise ValueError(f"variable_type must be one of {allowed}, got '{v}'")
        return v


class CausalEdge(BaseModel):
    """
    A directed causal relationship between two economic variables.

    Attributes
    ----------
    cause : str
        Node ``id`` of the causal ancestor.
    effect : str
        Node ``id`` of the causal descendant.
    confounder : str | None
        Node ``id`` of a third variable that jointly causes both ``cause``
        and ``effect``, if one exists in the graph. ``None`` otherwise.
    mechanism : str
        Plain-English description of the causal transmission channel.
    confidence_score : float
        Model-assigned confidence that this causal edge is supported by the
        source passage, on a scale of 0.0 (none) to 1.0 (certain).
    edge_sign : str
        Direction of the causal effect: ``"positive"`` (cause increases effect),
        ``"negative"`` (cause decreases effect), or ``"ambiguous"``.
    """

    cause: str = Field(..., description="Node id of the cause variable.")
    effect: str = Field(..., description="Node id of the effect variable.")
    confounder: str | None = Field(
        default=None,
        description="Node id of a confounding variable, if any; else null.",
    )
    mechanism: str = Field(
        ..., description="Description of the causal transmission channel."
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for this causal link, between 0.0 and 1.0.",
    )
    edge_sign: str = Field(
        ...,
        description="Direction of the effect: 'positive', 'negative', or 'ambiguous'.",
    )

    @field_validator("edge_sign")
    @classmethod
    def validate_edge_sign(cls, v: str) -> str:
        allowed = {"positive", "negative", "ambiguous"}
        if v not in allowed:
            raise ValueError(f"edge_sign must be one of {allowed}, got '{v}'")
        return v

    @field_validator("confidence_score")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"confidence_score must be in [0, 1], got {v}")
        return round(v, 4)


class CausalDAG(BaseModel):
    """
    A complete Causal Directed Acyclic Graph extracted from a macroeconomic text.

    Attributes
    ----------
    source_text_snippet : str
        First 200 characters of the source paragraph for traceability.
    nodes : list[CausalNode]
        All economic variable nodes present in the DAG.
    edges : list[CausalEdge]
        All directed causal edges. Every ``cause`` and ``effect`` id must
        reference a node in ``nodes``.
    summary : str
        One-paragraph narrative summary of the key causal dynamics identified.
    """

    source_text_snippet: str = Field(
        ..., description="First 200 characters of the source paragraph."
    )
    nodes: list[CausalNode] = Field(..., description="Economic variable nodes.")
    edges: list[CausalEdge] = Field(..., description="Directed causal edges.")
    summary: str = Field(
        ..., description="Narrative summary of the principal causal dynamics."
    )

    @field_validator("edges")
    @classmethod
    def validate_edge_node_refs(cls, edges: list[CausalEdge], info: Any) -> list[CausalEdge]:
        """Ensure every edge cause/effect references a declared node id."""
        # info.data only populated after 'nodes' is validated; guard against missing key
        nodes_data: list[CausalNode] = info.data.get("nodes", [])
        node_ids = {n.id for n in nodes_data}
        for edge in edges:
            if edge.cause not in node_ids:
                raise ValueError(
                    f"Edge cause '{edge.cause}' does not match any declared node id."
                )
            if edge.effect not in node_ids:
                raise ValueError(
                    f"Edge effect '{edge.effect}' does not match any declared node id."
                )
            if edge.confounder is not None and edge.confounder not in node_ids:
                raise ValueError(
                    f"Edge confounder '{edge.confounder}' does not match any declared node id."
                )
        return edges


# ---------------------------------------------------------------------------
# 3. Tool schema (Anthropic tool-calling format)
# ---------------------------------------------------------------------------

_DAG_TOOL: dict[str, Any] = {
    "name": "return_causal_dag",
    "description": (
        "Extract a structured Causal DAG from a macroeconomic text passage. "
        "Identify all economic variable nodes and the directed causal edges between them, "
        "including any confounders, transmission mechanisms, and confidence scores."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "source_text_snippet": {
                "type": "string",
                "description": "First 200 characters of the source paragraph.",
            },
            "nodes": {
                "type": "array",
                "description": "List of economic variable nodes.",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "label": {"type": "string"},
                        "description": {"type": "string"},
                        "variable_type": {
                            "type": "string",
                            "enum": ["policy", "price", "real", "financial", "expectational", "external"],
                        },
                    },
                    "required": ["id", "label", "description", "variable_type"],
                },
            },
            "edges": {
                "type": "array",
                "description": "List of directed causal edges.",
                "items": {
                    "type": "object",
                    "properties": {
                        "cause": {"type": "string"},
                        "effect": {"type": "string"},
                        "confounder": {"type": ["string", "null"]},
                        "mechanism": {"type": "string"},
                        "confidence_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "edge_sign": {
                            "type": "string",
                            "enum": ["positive", "negative", "ambiguous"],
                        },
                    },
                    "required": ["cause", "effect", "confounder", "mechanism", "confidence_score", "edge_sign"],
                },
            },
            "summary": {
                "type": "string",
                "description": "Narrative summary of the principal causal dynamics.",
            },
        },
        "required": ["source_text_snippet", "nodes", "edges", "summary"],
    },
}

# ---------------------------------------------------------------------------
# 4. Core extraction function
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert macroeconomist and causal-inference specialist. Your task is to read \
a passage of macroeconomic text—such as central bank minutes, research papers, or \
policy reports—and extract a precise Causal Directed Acyclic Graph (DAG).

Rules:
- Every node must represent a distinct, named economic variable present or clearly implied \
  in the passage.
- Every edge must be grounded in explicit or strongly implied causal language in the passage \
  (e.g., "caused", "led to", "transmitted to", "compressed", "reignited").
- Confounders must be variables already declared as nodes.
- Confidence scores must reflect how explicitly the causal link is stated (1.0 = direct \
  assertion, 0.5 = inference, 0.2 = weak implication).
- Do NOT invent variables or edges not supported by the passage.
- Call the `return_causal_dag` tool with your complete structured output.
"""


def extract_causal_dag(
    paragraph: str,
    *,
    model: str = "claude-sonnet-4-5",
    api_key: str | None = None,
) -> CausalDAG:
    """
    Parse a macroeconomic paragraph and return a validated CausalDAG object.

    Uses the Anthropic tool-calling API to coerce Claude into returning JSON
    that strictly conforms to the CausalDAG Pydantic schema.

    Parameters
    ----------
    paragraph : str
        Raw macroeconomic prose to analyse.
    model : str
        Anthropic model identifier. Defaults to ``"claude-sonnet-4-5"``.
    api_key : str | None
        Anthropic API key. Falls back to ``ANTHROPIC_API_KEY`` env var if ``None``.

    Returns
    -------
    CausalDAG
        A fully validated causal DAG with nodes, edges, and summary.

    Raises
    ------
    anthropic.APIError
        On any Anthropic API communication failure.
    ValueError
        If Claude does not invoke the tool or returns malformed data.
    pydantic.ValidationError
        If the returned JSON fails schema validation.
    """
    client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=_SYSTEM_PROMPT,
        tools=[_DAG_TOOL],
        tool_choice={"type": "any"},  # force tool use
        messages=[
            {
                "role": "user",
                "content": (
                    "Extract the causal DAG from the following macroeconomic passage:\n\n"
                    f"{paragraph}"
                ),
            }
        ],
    )

    # Locate the tool_use block
    tool_use_block = next(
        (block for block in response.content if block.type == "tool_use"),
        None,
    )
    if tool_use_block is None:
        raise ValueError(
            "Claude did not invoke the `return_causal_dag` tool. "
            f"Response stop_reason: {response.stop_reason}. "
            f"Content: {response.content}"
        )

    raw_payload: dict[str, Any] = tool_use_block.input  # already a dict from SDK

    # Validate and coerce into Pydantic model
    dag = CausalDAG.model_validate(raw_payload)
    return dag


# ---------------------------------------------------------------------------
# 5. Batch extraction helper
# ---------------------------------------------------------------------------


def batch_extract(
    paragraphs: list[str],
    *,
    model: str = "claude-sonnet-4-5",
    api_key: str | None = None,
) -> list[CausalDAG]:
    """
    Extract causal DAGs from multiple paragraphs sequentially.

    Parameters
    ----------
    paragraphs : list[str]
        List of macroeconomic prose passages to process.
    model : str
        Anthropic model identifier.
    api_key : str | None
        Anthropic API key; falls back to ``ANTHROPIC_API_KEY`` env var.

    Returns
    -------
    list[CausalDAG]
        One CausalDAG per input paragraph, in order.
    """
    results: list[CausalDAG] = []
    for i, paragraph in enumerate(paragraphs, start=1):
        print(f"[extractor] Processing paragraph {i}/{len(paragraphs)} …")
        dag = extract_causal_dag(paragraph, model=model, api_key=api_key)
        results.append(dag)
        print(
            f"[extractor]   → {len(dag.nodes)} nodes, {len(dag.edges)} edges extracted."
        )
    return results


# ---------------------------------------------------------------------------
# 6. CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    target = MOCK_PARAGRAPHS[0] if len(sys.argv) < 2 else MOCK_PARAGRAPHS[int(sys.argv[1])]
    dag = extract_causal_dag(target)
    print(dag.model_dump_json(indent=2))
