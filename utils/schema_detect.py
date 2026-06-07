"""Schema auto-discovery: map any tabular dataset's columns to semantic concepts.

Outputs schema_map.json so all downstream agents are dataset-agnostic.
All detection is deterministic (zero LLM calls) — name heuristics + statistical tests.

Supported concepts:
  record_id     — unique row identifier
  part_number   — alphanumeric SKU / product code
  customer_id   — external entity reference (validated against a lookup)
  production_date
  ship_date
  quantity      — integer count
  weight        — numeric float with physical units
  status        — low-cardinality categorical (order state)
  plant         — optional facility / plant identifier
"""

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

OUTPUT_DIR = Path(__file__).parent.parent / "output"

# ── Name-pattern hints (case-insensitive regex, scored) ──────────────────────
_CONCEPT_HINTS: dict[str, list[str]] = {
    "record_id":        [r"record.?id", r"row.?id", r"\bid\b", r"^id$", r"uuid", r"key"],
    "part_number":      [r"part.?num", r"part.?no", r"sku", r"item.?code", r"product.?code", r"pn"],
    "customer_id":      [r"customer.?id", r"cust.?id", r"client.?id", r"account.?id"],
    "production_date":  [r"prod.?date", r"manufacture.?date", r"made.?date", r"mfg.?date", r"build.?date"],
    "ship_date":        [r"ship.?date", r"delivery.?date", r"dispatch.?date", r"shipped"],
    "quantity":         [r"qty", r"quantity", r"count", r"units", r"amount", r"num"],
    "weight":           [r"weight", r"mass", r"kg", r"lbs", r"grams"],
    "status":           [r"status", r"state", r"stage", r"phase", r"condition"],
    "plant":            [r"plant", r"facility", r"warehouse", r"location", r"site"],
}


@dataclass
class SchemaMap:
    record_id:        Optional[str] = None
    part_number:      Optional[str] = None
    customer_id:      Optional[str] = None
    production_date:  Optional[str] = None
    ship_date:        Optional[str] = None
    quantity:         Optional[str] = None
    weight:           Optional[str] = None
    status:           Optional[str] = None
    plant:            Optional[str] = None
    # Any extra columns not mapped to a concept
    unmapped:         list[str] = field(default_factory=list)
    # Human-readable confidence notes
    notes:            dict[str, str] = field(default_factory=dict)

    def get(self, concept: str, fallback: Optional[str] = None) -> Optional[str]:
        return getattr(self, concept, fallback) or fallback

    def require(self, concept: str) -> str:
        val = getattr(self, concept, None)
        if val is None:
            raise ValueError(f"Schema: concept '{concept}' was not detected in this dataset.")
        return val


def _name_score(col: str, hints: list[str]) -> int:
    col_lower = col.lower()
    return sum(1 for h in hints if re.search(h, col_lower))


def _is_datetime_col(series: pd.Series) -> bool:
    sample = series.dropna().head(50)
    if sample.empty:
        return False
    try:
        pd.to_datetime(sample, errors="raise")
        return True
    except Exception:
        return False


def _is_unique_id(series: pd.Series) -> bool:
    return series.nunique() == len(series.dropna())


def detect_schema(df: pd.DataFrame) -> SchemaMap:
    """Infer semantic column roles from column names and data statistics."""
    schema = SchemaMap()
    assigned: set[str] = set()

    # ── Step 1: score every column against every concept by name ─────────────
    scores: dict[str, dict[str, int]] = {
        concept: {col: _name_score(col, hints) for col in df.columns}
        for concept, hints in _CONCEPT_HINTS.items()
    }

    def _best(concept: str, exclude: set[str] | None = None) -> Optional[str]:
        """Pick the highest-scoring unassigned column for a concept."""
        exc = exclude or set()
        candidates = {col: s for col, s in scores[concept].items()
                      if col not in assigned and col not in exc and s > 0}
        return max(candidates, key=candidates.__getitem__) if candidates else None

    # ── Step 2: assign via heuristics + statistical confirmation ─────────────

    # record_id: unique string column with high name score
    for col in df.columns:
        if _name_score(col, _CONCEPT_HINTS["record_id"]) > 0 and _is_unique_id(df[col]):
            schema.record_id = col
            assigned.add(col)
            schema.notes["record_id"] = f"Unique values={df[col].nunique()}"
            break
    if not schema.record_id:
        # Fallback: first all-unique column
        for col in df.columns:
            if _is_unique_id(df[col]):
                schema.record_id = col
                assigned.add(col)
                schema.notes["record_id"] = "Fallback: first all-unique column"
                break

    # date columns: must be parseable as datetime
    for concept in ("production_date", "ship_date"):
        col = _best(concept, assigned)
        if col and _is_datetime_col(df[col]):
            setattr(schema, concept, col)
            assigned.add(col)
            schema.notes[concept] = "Name match + datetime-parseable"
        else:
            # Search all remaining columns for datetime shape
            for c in df.columns:
                if c not in assigned and _is_datetime_col(df[c]):
                    setattr(schema, concept, c)
                    assigned.add(c)
                    schema.notes[concept] = f"Datetime-parseable fallback (no name match): {c}"
                    break

    # quantity: integer-typed numeric with name match
    col = _best("quantity", assigned)
    if col and pd.api.types.is_numeric_dtype(df[col]):
        schema.quantity = col
        assigned.add(col)
        schema.notes["quantity"] = f"Numeric, name match. Range: {df[col].min()}–{df[col].max()}"
    else:
        for c in df.columns:
            if c not in assigned and pd.api.types.is_integer_dtype(df[c]):
                schema.quantity = c
                assigned.add(c)
                schema.notes["quantity"] = f"Integer fallback: {c}"
                break

    # weight: float numeric with name match; if not, find the float column with highest variance
    col = _best("weight", assigned)
    if col and pd.api.types.is_float_dtype(df[col]):
        schema.weight = col
        assigned.add(col)
        schema.notes["weight"] = f"Float, name match. Median={df[col].median():.2f}"
    else:
        # Fallback: numeric float column with highest coefficient of variation (likely weight)
        float_cols = [c for c in df.columns if c not in assigned
                      and pd.api.types.is_float_dtype(df[c]) and df[c].mean() != 0]
        if float_cols:
            col = max(float_cols, key=lambda c: df[c].std() / df[c].mean() if df[c].mean() else 0)
            schema.weight = col
            assigned.add(col)
            schema.notes["weight"] = f"Float fallback (highest CV): {col}"

    # status: low-cardinality string column
    col = _best("status", assigned)
    if col and df[col].nunique() <= 15:
        schema.status = col
        assigned.add(col)
        schema.notes["status"] = f"Cardinality={df[col].nunique()}: {sorted(df[col].dropna().unique()[:8].tolist())}"
    else:
        for c in df.columns:
            if c not in assigned and df[c].dtype == object and df[c].nunique() <= 10:
                schema.status = c
                assigned.add(c)
                schema.notes["status"] = f"Low-cardinality string fallback: {c}"
                break

    # part_number, customer_id, plant: string columns with name match
    for concept in ("part_number", "customer_id", "plant"):
        col = _best(concept, assigned)
        if col:
            setattr(schema, concept, col)
            assigned.add(col)
            schema.notes[concept] = f"Name match. Cardinality={df[col].nunique()}"

    schema.unmapped = [c for c in df.columns if c not in assigned]
    return schema


def save_schema(schema: SchemaMap) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / "schema_map.json"
    with open(path, "w") as f:
        json.dump(asdict(schema), f, indent=2)
    return path


def load_schema() -> SchemaMap:
    path = OUTPUT_DIR / "schema_map.json"
    if not path.exists():
        raise FileNotFoundError("schema_map.json not found — run detect_schema() first")
    with open(path) as f:
        data = json.load(f)
    data.pop("unmapped", None)
    data.pop("notes", None)
    return SchemaMap(**{k: v for k, v in data.items() if k in SchemaMap.__dataclass_fields__})
