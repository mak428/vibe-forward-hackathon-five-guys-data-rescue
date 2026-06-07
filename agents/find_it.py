"""Agent 1 — Find It (data-agnostic).

Reads ANY tabular dataset, profiles its columns into roles (id / numeric / date /
categorical / identifier / foreign-key), then runs schema-agnostic detectors for
the universal data-quality problem classes:

    exact duplicates · near-duplicate (case/whitespace) variants · format /
    separator drift in identifiers · orphaned foreign-key references ·
    numeric sign anomalies · robust numeric outliers (decimal/unit errors) ·
    temporal order violations (e.g. ship-before-production, discovered from data)

No column names, prefixes, thresholds, or row counts are hardcoded — everything is
inferred from the data. A thin compatibility layer re-labels findings to the legacy
Harven subtypes when those exact columns are present, so the existing Agent 2/3
pipeline keeps working unchanged; on any other dataset the generic labels flow through.

Findings are written to Cognee (vector ingestion, zero LLM calls) and to
output/findings.json as a deterministic fallback.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import cognee
except Exception:  # keep Agent 1 runnable even if cognee isn't installed/configured
    cognee = None

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "output"

# Tunable (statistical, not domain) thresholds.
NEAR_DUP_NORM = True          # normalise case/whitespace when hunting near-dups
SIGN_POSITIVE_FRAC = 0.95     # col is "should be positive" if >=95% of values are >0
DATE_RULE_FRAC = 0.98         # a date-order rule must hold for >=98% of row pairs
MAD_Z = 6.0                   # robust-outlier z cutoff
OUTLIER_RATIO = 4.0           # AND value must be >4x the group median
MIN_GROUP = 4                 # need >=4 members in a group to judge an outlier
GROUP_TIGHTNESS = 0.5         # only trust group-outliers when typical group MAD/median < this
TEXT_TOKEN_SPACE_FRAC = 0.30  # >30% of values contain a space => free text, not a code

ID_NAME_HINT = re.compile(r"(^id$|_id$|^id_|record|uuid|guid|key$|_key$|number$|code$)", re.I)
DATE_LIKE = re.compile(r"^\s*\d{1,4}[-/]\d{1,2}[-/]\d{1,4}")


# --------------------------------------------------------------------------- #
# Schema profiling
# --------------------------------------------------------------------------- #
def _is_numeric(s: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(s):
        return True
    coerced = pd.to_numeric(s, errors="coerce")
    return s.notna().any() and coerced.notna().mean() >= 0.9


def _is_date(s: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(s):
        return False
    sample = s.dropna().astype(str)
    if sample.empty:
        return False
    if sample.str.match(DATE_LIKE).mean() < 0.7:
        return False
    parsed = pd.to_datetime(s, errors="coerce")
    return parsed.notna().mean() >= 0.8


def _is_codelike(s: pd.Series) -> bool:
    """A short identifier/code column (part numbers, IDs) — not free text or a sentence."""
    vals = s.dropna().astype(str)
    if vals.empty:
        return False
    has_space = vals.str.contains(r"\s").mean()
    return has_space < TEXT_TOKEN_SPACE_FRAC


def profile_schema(df: pd.DataFrame, ref_tables: dict[str, pd.DataFrame]) -> dict:
    n = len(df)
    cols = list(df.columns)

    # id column: prefer a unique, id-named column; else the highest-cardinality unique col.
    unique_cols = [c for c in cols if df[c].is_unique]
    id_named = [c for c in unique_cols if ID_NAME_HINT.search(c)]
    if id_named:
        id_col = max(id_named, key=lambda c: df[c].nunique())
    elif unique_cols:
        id_col = max(unique_cols, key=lambda c: df[c].nunique())
    else:  # no perfectly-unique column; fall back to most-distinct id-named or most-distinct col
        cand = [c for c in cols if ID_NAME_HINT.search(c)] or cols
        id_col = max(cand, key=lambda c: df[c].nunique())

    numeric, dates, categorical, identifier = [], [], [], []
    for c in cols:
        if c == id_col:
            continue
        s = df[c]
        if _is_date(s):
            dates.append(c)
        elif _is_numeric(s):
            numeric.append(c)
        else:
            card = s.nunique(dropna=True)
            if _is_codelike(s):
                identifier.append(c)
            # categorical = low-cardinality grouping key (codes count too)
            if card >= 2 and card <= max(2, n * 0.6):
                categorical.append(c)

    # foreign keys: a column whose values reference a unique key in some companion table
    fks: dict[str, dict] = {}
    for fk_col in cols:
        if fk_col == id_col or fk_col in dates or fk_col in numeric:
            continue
        prim_vals = set(df[fk_col].dropna().astype(str))
        if not prim_vals:
            continue
        for ref_name, ref in ref_tables.items():
            for ref_col in ref.columns:
                if not ref[ref_col].is_unique:
                    continue
                ref_vals = set(ref[ref_col].dropna().astype(str))
                name_match = fk_col == ref_col
                overlap = len(prim_vals & ref_vals) / len(prim_vals)
                if name_match or overlap >= 0.5:
                    fks[fk_col] = {"ref_table": ref_name, "ref_col": ref_col, "ref_values": ref_vals}
                    break
            if fk_col in fks:
                break

    return {
        "n": n, "id": id_col, "dates": dates, "numeric": numeric,
        "categorical": categorical, "identifier": identifier, "fks": fks,
    }


def _norm(series: pd.Series) -> pd.Series:
    """Canonicalise an identifier: drop [-_ ] and uppercase. BOLT_103 == bolt-103."""
    return series.astype(str).str.replace(r"[-_\s]", "", regex=True).str.upper()


# --------------------------------------------------------------------------- #
# Generic detectors  — each returns a list of finding dicts.
# --------------------------------------------------------------------------- #
def detect_exact_duplicates(df, p):
    idc = p["id"]
    cols_no_id = [c for c in df.columns if c != idc]
    extras = df[df.duplicated(subset=cols_no_id, keep="first")]
    if extras.empty:
        return []
    return [{
        "type": "duplicate", "subtype": "exact_duplicate",
        "columns": cols_no_id, "count": int(len(extras)),
        "record_ids": extras[idc].astype(str).tolist(),
        "severity": "Medium",
        "description": (
            f"{len(extras)} records are exact copies of another record "
            f"(identical on every column except '{idc}') — duplicate entries."
        ),
        "evidence": [f"{len(extras)} redundant copies across {df.duplicated(subset=cols_no_id, keep=False).sum()} rows"],
    }]


def detect_near_duplicates(df, p):
    """Rows that duplicate another after normalising case/whitespace in string cols,
    but are NOT exact duplicates — i.e. same record re-entered with formatting drift."""
    idc = p["id"]
    cols_no_id = [c for c in df.columns if c != idc]
    norm = df[cols_no_id].copy()
    for c in cols_no_id:
        if norm[c].dtype == object:
            norm[c] = norm[c].astype(str).str.strip().str.upper()
    exact = df.duplicated(subset=cols_no_id, keep=False)
    near = norm.duplicated(keep=False) & ~exact
    hits = df[near]
    if hits.empty:
        return []
    return [{
        "type": "naming_conflict", "subtype": "near_duplicate_variant",
        "columns": cols_no_id, "count": int(len(hits)),
        "record_ids": hits[idc].astype(str).tolist(),
        "severity": "Medium",
        "description": (
            f"{len(hits)} records match another record once case/whitespace is normalised "
            f"but differ in raw formatting — near-duplicate variants from inconsistent entry."
        ),
        "evidence": [f"{int(near.sum())} rows collapse to a duplicate after case/space normalisation"],
    }]


def detect_format_drift(df, p):
    """For identifier/code columns: one canonical value written with >1 raw spelling.
    Splits the deviants into structural separator drift (→ unit-format drift) vs
    case/whitespace-only variation (→ near-duplicate variants)."""
    idc = p["id"]
    out = []
    for c in p["identifier"]:
        if c == idc:
            continue
        raw = df[c].astype(str)
        key = _norm(raw)
        n_spell = raw.groupby(key).transform("nunique")
        # canonical = most common raw spelling per key; deviants = the rest
        mode_per_key = raw.groupby(key).transform(
            lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]
        )
        deviant = (n_spell > 1) & (raw != mode_per_key)
        if not deviant.any():
            continue
        # case/whitespace-only difference vs structural (separator) difference
        casews_only = deviant & (raw.str.strip().str.upper() == mode_per_key.str.strip().str.upper())
        separator = deviant & ~casews_only
        n_keys = int(key[deviant].nunique())
        if separator.any():
            out.append({
                "type": "naming_conflict", "subtype": "separator_drift",
                "columns": [c], "count": int(separator.sum()),
                "record_ids": df.loc[separator, idc].astype(str).tolist(),
                "severity": "Medium",
                "description": (
                    f"{int(separator.sum())} values in '{c}' use a different separator/structure "
                    f"than the canonical spelling of the same identifier (e.g. underscore vs hyphen) "
                    f"across {n_keys} identifiers — a systematic format drift that splits joins."
                ),
                "evidence": [f"{n_keys} canonical keys in '{c}' written with mixed separators"],
            })
        if casews_only.any():
            out.append({
                "type": "naming_conflict", "subtype": "format_variant",
                "columns": [c], "count": int(casews_only.sum()),
                "record_ids": df.loc[casews_only, idc].astype(str).tolist(),
                "severity": "Medium",
                "description": (
                    f"{int(casews_only.sum())} values in '{c}' differ from the canonical spelling "
                    f"only by case/whitespace — near-duplicate identifier variants."
                ),
                "evidence": [f"case/whitespace-only variants in '{c}'"],
            })
    return out


def detect_orphans(df, p):
    idc = p["id"]
    out = []
    for fk_col, info in p["fks"].items():
        vals = df[fk_col].astype(str)
        orphan_mask = ~vals.isin(info["ref_values"])
        if not orphan_mask.any():
            continue
        unknown = sorted(vals[orphan_mask].unique())
        out.append({
            "type": "orphaned_reference", "subtype": "unknown_reference",
            "columns": [fk_col], "count": int(orphan_mask.sum()),
            "record_ids": df.loc[orphan_mask, idc].astype(str).tolist(),
            "unknown_ids": unknown,
            "severity": "High",
            "description": (
                f"{int(orphan_mask.sum())} records reference '{fk_col}' values "
                f"({len(unknown)} distinct) absent from '{info['ref_table']}.{info['ref_col']}' — "
                f"orphaned references with no master record."
            ),
            "evidence": [f"{len(unknown)} unknown ids, e.g. {unknown[:5]}"],
        })
    return out


def detect_sign_anomaly(df, p):
    idc = p["id"]
    out = []
    for c in p["numeric"]:
        s = pd.to_numeric(df[c], errors="coerce")
        nn = s.dropna()
        if nn.empty:
            continue
        if (nn > 0).mean() >= SIGN_POSITIVE_FRAC and (nn <= 0).any():
            mask = s <= 0
            out.append({
                "type": "impossible_value", "subtype": "non_positive_value",
                "columns": [c], "count": int(mask.sum()),
                "record_ids": df.loc[mask.fillna(False), idc].astype(str).tolist(),
                "severity": "Medium",
                "description": (
                    f"{int(mask.sum())} records have '{c}' <= 0 while the column is otherwise "
                    f"almost entirely positive — physically impossible / sign error."
                ),
                "evidence": [f"values: {sorted(s[mask].dropna().unique().tolist())[:8]}"],
            })
    return out


def _best_group_col(df, p, numeric_col):
    """Pick the most granular eligible grouping column for a numeric column."""
    n = p["n"]
    candidates = []
    for c in set(p["identifier"] + p["categorical"]):
        if c == numeric_col:
            continue
        card = df[c].nunique(dropna=True)
        if 2 <= card and n / max(card, 1) >= MIN_GROUP:
            candidates.append((card, c))
    if not candidates:
        return None
    return max(candidates)[1]  # highest cardinality -> most specific grouping


def detect_outliers(df, p):
    idc = p["id"]
    out = []
    for c in p["numeric"]:
        s = pd.to_numeric(df[c], errors="coerce")
        gcol = _best_group_col(df, p, c)
        if gcol is None:
            continue
        gkey = _norm(df[gcol]) if gcol in p["identifier"] else df[gcol].astype(str)
        groups = {k: s.loc[idx].dropna() for k, idx in s.groupby(gkey).groups.items()}
        groups = {k: x for k, x in groups.items() if len(x) >= MIN_GROUP and x.median() != 0}
        if not groups:
            continue
        # Tightness gate: only trust outliers when the grouping actually explains the
        # numeric (groups are internally tight). Drops naturally-variable cols like
        # order quantity while keeping physically-constrained ones like weight.
        rel_spread = np.median([(x - x.median()).abs().median() / abs(x.median()) for x in groups.values()])
        if rel_spread > GROUP_TIGHTNESS:
            continue
        flagged_idx = []
        for x in groups.values():
            med = x.median()
            mad = (x - med).abs().median()
            if mad == 0:
                continue
            z = (x - med).abs() / (1.4826 * mad)
            ratio = x / med
            flagged_idx += list(x[(z > MAD_Z) & (ratio > OUTLIER_RATIO)].index)
        if not flagged_idx:
            continue
        out.append({
            "type": "impossible_value", "subtype": "numeric_outlier",
            "columns": [c], "count": int(len(flagged_idx)),
            "record_ids": df.loc[flagged_idx, idc].astype(str).tolist(),
            "severity": "High",
            "description": (
                f"{len(flagged_idx)} records have '{c}' as a robust outlier (>6 MAD and >4x median) "
                f"within their '{gcol}' group — consistent with a decimal-shift or unit error."
            ),
            "evidence": [f"grouped by '{gcol}'; MAD z>{MAD_Z} and ratio>{OUTLIER_RATIO}x median"],
        })
    return out


def detect_date_order(df, p):
    """Discover a consistent ordering rule between any two date columns and flag violators."""
    idc = p["id"]
    dts = {c: pd.to_datetime(df[c], errors="coerce") for c in p["dates"]}
    out = []
    cols = p["dates"]
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            da, db = dts[a], dts[b]
            both = da.notna() & db.notna()
            if both.sum() < 20:
                continue
            le = (da[both] <= db[both]).mean()
            if le >= DATE_RULE_FRAC:
                rule, viol = f"{a} <= {b}", both & (da > db)
            elif le <= (1 - DATE_RULE_FRAC):
                rule, viol = f"{b} <= {a}", both & (db > da)
            else:
                continue
            if not viol.any():
                continue
            out.append({
                "type": "impossible_value", "subtype": "date_order_violation",
                "columns": [a, b], "count": int(viol.sum()),
                "record_ids": df.loc[viol, idc].astype(str).tolist(),
                "severity": "High",
                "description": (
                    f"{int(viol.sum())} records violate the learned rule '{rule}' "
                    f"(which holds for {max(le, 1 - le):.1%} of rows) — impossible chronology."
                ),
                "evidence": [f"rule '{rule}' inferred from data; {int(viol.sum())} violations"],
            })
    return out


# --------------------------------------------------------------------------- #
# The 6 canonical data-quality classes every finding is classified into.
# Generic archetypes — they hold for any tabular dataset, not just Harven.
# --------------------------------------------------------------------------- #
CANON_CLASSES = [
    ("exact_duplicates",        "Exact duplicates"),
    ("near_duplicate_variants", "Near-duplicate variants (case/whitespace)"),
    ("unit_format_drift",       "Unit-format drift"),
    ("orphaned_references",     "Orphaned references"),
    ("decimal_shift",           "Decimal-shift / unit-error values"),
    ("impossible_values",       "Impossible values"),
]
CLASS_LABEL = dict(CANON_CLASSES)

# Maps each detector's generic subtype onto one of the 6 canonical classes.
CLASS_OF = {
    "exact_duplicate":        "exact_duplicates",
    "near_duplicate_variant": "near_duplicate_variants",
    "format_variant":         "near_duplicate_variants",   # identifier case/whitespace drift
    "separator_drift":        "unit_format_drift",
    "unknown_reference":      "orphaned_references",
    "numeric_outlier":        "decimal_shift",
    "non_positive_value":     "impossible_values",
    "date_order_violation":   "impossible_values",
}


def assign_classes(findings: list[dict]) -> list[dict]:
    for f in findings:
        ckey = CLASS_OF.get(f["subtype"], "impossible_values")
        f["class"] = ckey
        f["class_label"] = CLASS_LABEL[ckey]
    return findings


# --------------------------------------------------------------------------- #
# Legacy compatibility — re-label generic findings to Harven subtypes so the
# existing Agent 2/3 pipeline dispatches exactly as before. Pure column-name
# match; on any other dataset this is a no-op and generic labels pass through.
# --------------------------------------------------------------------------- #
def apply_legacy_aliases(findings: list[dict], p: dict) -> list[dict]:
    cols = set(p["dates"] + p["numeric"] + p["categorical"] + p["identifier"] + list(p["fks"]))
    harven = {"part_number", "customer_id", "production_date", "ship_date", "weight_kg", "quantity"} <= cols
    if not harven:
        return findings
    for f in findings:
        st, fc = f["subtype"], f.get("columns", [])
        if st == "exact_duplicate":
            f["subtype"] = "semantic_duplicate"
        elif st == "near_duplicate_variant":
            f["subtype"] = "malformed_part_number"
        elif st == "separator_drift" and fc == ["part_number"]:
            f["subtype"] = "separator_mismatch"
        elif st == "format_variant" and fc == ["part_number"]:
            f["subtype"] = "malformed_part_number"
        elif st == "unknown_reference" and fc == ["customer_id"]:
            f["subtype"] = "unknown_customer_id"
        elif st == "numeric_outlier" and fc == ["weight_kg"]:
            f["subtype"] = "weight_inflation"
        elif st == "non_positive_value" and fc == ["quantity"]:
            f["subtype"] = "non_positive_quantity"
        elif st == "date_order_violation" and set(fc) == {"production_date", "ship_date"}:
            f["subtype"] = "ship_before_production"
    return findings


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _load_tables() -> tuple[pd.DataFrame, dict[str, pd.DataFrame], str]:
    """Largest CSV in data/ = primary table; the rest = reference tables."""
    csvs = sorted(DATA_DIR.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")
    # keep_default_na=False so legit values like "NA" (North America), "None", "null"
    # are NOT silently turned into missing; only genuinely blank cells become "".
    frames = {f.name: pd.read_csv(f, dtype=str, keep_default_na=False) for f in csvs}
    primary_name = max(frames, key=lambda k: len(frames[k]))
    primary = frames[primary_name]
    refs = {k: v for k, v in frames.items() if k != primary_name}
    return primary, refs, primary_name


async def find_issues() -> list[dict]:
    OUTPUT_DIR.mkdir(exist_ok=True)
    df, ref_tables, primary_name = _load_tables()
    p = profile_schema(df, ref_tables)

    findings: list[dict] = []
    for detector in (
        detect_exact_duplicates, detect_near_duplicates, detect_format_drift,
        detect_orphans, detect_sign_anomaly, detect_outliers, detect_date_order,
    ):
        findings += detector(df, p)

    findings = assign_classes(findings)        # tag every finding with 1 of 6 classes
    findings = apply_legacy_aliases(findings, p)

    sev_rank = {"High": 0, "Medium": 1, "Low": 2}
    findings.sort(key=lambda f: (sev_rank.get(f["severity"], 3), -f["count"]))

    total_affected = sum(f["count"] for f in findings)

    # Roll up into the 6 canonical classes (ordered).
    class_rollup = {
        label: sum(f["count"] for f in findings if f["class"] == key)
        for key, label in CANON_CLASSES
    }

    # ── Write findings.json (list — inter-agent contract) + classified rollup ──
    with open(OUTPUT_DIR / "findings.json", "w", encoding="utf-8") as fh:
        json.dump(findings, fh, indent=2, default=str)
    with open(OUTPUT_DIR / "findings_by_class.json", "w", encoding="utf-8") as fh:
        grouped = {
            label: [f for f in findings if f["class"] == key]
            for key, label in CANON_CLASSES
        }
        json.dump(
            {"class_summary": class_rollup, "total_affected": total_affected, "by_class": grouped},
            fh, indent=2, default=str,
        )

    # ── Cognee (vector ingestion only — zero LLM calls); guarded ──
    profile_line = (
        f"Profiled '{primary_name}' ({p['n']} rows): id='{p['id']}', "
        f"dates={p['dates']}, numeric={p['numeric']}, identifiers={p['identifier']}, "
        f"foreign_keys={list(p['fks'])}."
    )
    class_line = "; ".join(f"{label}={n}" for label, n in class_rollup.items())
    summary = (
        f"Agent 1 (Find It) scanned {p['n']} records. {profile_line} "
        f"Classified {total_affected} affected records into 6 data-quality classes: "
        f"{class_line}."
    )
    cognee_written = False
    if cognee is not None:
        try:
            await cognee.add(summary, dataset_name="data_rescue")
            # one doc per canonical class so Agent 2+ can recall by class …
            for key, label in CANON_CLASSES:
                cfind = [f for f in findings if f["class"] == key]
                if not cfind:
                    continue
                await cognee.add(
                    f"CLASS [{label}] total={sum(f['count'] for f in cfind)} "
                    f"across {len(cfind)} finding(s): "
                    + " | ".join(f"{f['subtype']} cols={f.get('columns')} n={f['count']}" for f in cfind),
                    dataset_name="data_rescue",
                )
            # … and one doc per individual finding (carrying its class).
            for f in findings:
                await cognee.add(
                    f"ISSUE class='{f['class_label']}' [{f['type']}/{f['subtype']}] "
                    f"count={f['count']} cols={f.get('columns')} severity={f['severity']}: "
                    f"{f['description']}",
                    dataset_name="data_rescue",
                )
            cognee_written = True
        except Exception as e:
            print(f"[Agent 1 — Find It] Cognee write skipped ({type(e).__name__}: {e}); "
                  f"findings.json is the source of truth.")

    print(f"[Agent 1 - Find It] {primary_name}: {len(findings)} findings, "
          f"{total_affected} affected records "
          f"({'Cognee+' if cognee_written else ''}findings.json)")
    print(f"  schema: id='{p['id']}' dates={p['dates']} numeric={p['numeric']} "
          f"identifiers={p['identifier']} fks={list(p['fks'])}")
    print("  -- findings classified into 6 classes --")
    for label, n in class_rollup.items():
        print(f"  > {label:42s} {n}")
    for f in findings:
        print(f"      [{f['severity']:6s}] {f['class_label'][:26]:26s} "
              f"{f['subtype']} cols={f.get('columns')}: {f['count']}")

    return findings