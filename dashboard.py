"""FD002 Engine Monitor — saved-prefix replay for the revised study.

Run locally with ``streamlit run dashboard.py``. This app only reads the
completed research run in ``results/engine_fd002`` and the supplied FD002 data.
It does not connect to an engine, issue work orders, or make maintenance calls.
"""

from __future__ import annotations

import importlib
import json
import math
import sys
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
RUN_DIR = ROOT / "results" / "engine_fd002"
PREDICTIONS = RUN_DIR / "predictions.csv"
if not PREDICTIONS.exists():
    PREDICTIONS = RUN_DIR / "prediction_partitions"
SENSORS = RUN_DIR / "sensors.csv"
SPLITS = RUN_DIR / "splits.json"

st.set_page_config(
    page_title="Engine Monitor · FD002",
    page_icon="✳",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown(
    """
    <style>
    :root {
      --ink:#15232c; --muted:#526772; --canvas:#f2f5f5; --paper:#ffffff;
      --navy:#12222d; --navy2:#1a303d; --line:#dce5e8; --cyan:#18a9bb;
      --cyan-deep:#087b8a; --amber:#ba7528; --red:#a84c45; --green:#31735f;
      --mono:ui-monospace, "SFMono-Regular", Consolas, monospace;
    }
    .stApp { background:var(--canvas); color:var(--ink); }
    .block-container { max-width:1640px; padding:3.1rem 2.2rem 3rem; }
    header[data-testid="stHeader"] { background:rgba(242,245,245,.92); }
    /* Remove implementation controls while keeping the header's sidebar toggle. */
    [data-testid="stAppDeployButton"], [data-testid="stMainMenuButton"],
    [data-testid="stStatusWidget"] { display:none!important; visibility:hidden!important; }
    [data-testid="stSidebar"] { background:var(--navy); border-right:1px solid #304451; }
    [data-testid="stSidebar"] * { color:#e8f0f2; }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color:#aebfc6!important; }
    [data-testid="stSidebar"] .brandline { color:#dce8eb!important; }
    [data-testid="stSidebar"] .brandmark { color:#66d1de; border-color:#66d1de; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label { padding:.24rem .18rem; }
    [data-testid="stSidebar"] [data-baseweb="select"] *, [data-testid="stSidebar"] [data-baseweb="input"] *, [data-testid="stSidebar"] input[role="combobox"] { color:#15232c!important; -webkit-text-fill-color:#15232c!important; }
    [data-testid="stSidebar"] input[role="combobox"] { background:#f7f9f9!important; }
    [data-testid="stSidebar"] input[role="combobox"]::placeholder { color:#526772!important; -webkit-text-fill-color:#526772!important; }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] { background:#f7f9f9; }
    [role="listbox"] *, [role="option"] { color:#15232c!important; }
    [data-testid="stSidebar"] [data-baseweb="popover"] * { color:#15232c!important; }
    [data-testid="stSidebar"] hr { border-color:#304451; }
    h1,h2,h3 { color:var(--ink); letter-spacing:-.035em; }
    h1 { font-size:2.3rem!important; font-weight:690!important; line-height:1.05!important; }
    h2 { font-size:1.32rem!important; font-weight:670!important; }
    h3 { font-size:1rem!important; font-weight:680!important; }
    p, label, [data-testid="stCaptionContainer"] { color:var(--muted); }
    [data-testid="stCaptionContainer"] p { color:#435963!important; }
    [data-testid="stMetric"] { padding:.15rem .75rem .4rem; border-left:2px solid var(--line); }
    [data-testid="stMetricLabel"] p { text-transform:uppercase; letter-spacing:.095em; font-size:.65rem; font-weight:700; }
    [data-testid="stMetricValue"] { color:var(--ink); font-family:var(--mono); font-size:1.45rem; font-variant-numeric:tabular-nums; }
    [data-testid="stTabs"] [role="tablist"] { gap:1rem; border-bottom:1px solid var(--line); }
    [data-testid="stTabs"] button[role="tab"] { font-weight:650; }
    [data-testid="stTabs"] button[aria-selected="true"] { color:var(--cyan-deep); border-bottom-color:var(--cyan); }
    [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:3px; }
    [data-testid="stExpander"] { background:var(--paper); border:1px solid var(--line); border-radius:3px; }
    .mast { display:flex; align-items:center; justify-content:space-between; gap:1rem; border-bottom:1px solid var(--line); padding:.2rem 0 .85rem; margin-bottom:1.2rem; }
    .brandline { display:flex; align-items:center; gap:.65rem; color:#27404c; font:700 .72rem var(--mono); letter-spacing:.12em; text-transform:uppercase; }
    .brandmark { width:25px; height:25px; border:2px solid var(--cyan); color:var(--cyan-deep); display:grid; place-items:center; font:700 .82rem var(--mono); transform:rotate(45deg); }
    .brandmark span { transform:rotate(-45deg); }
    .meta { color:#70818a; font:600 .68rem var(--mono); letter-spacing:.075em; text-transform:uppercase; }
    .research-strip { display:flex; flex-wrap:wrap; align-items:center; gap:.65rem; border:1px solid #cfe1e5; background:#e8f2f3; padding:.62rem .8rem; margin:.75rem 0 1.15rem; color:#264d57; font-size:.81rem; }
    .live-dot { width:.48rem; height:.48rem; display:inline-block; border-radius:50%; background:var(--cyan); box-shadow:0 0 0 3px rgba(24,169,187,.13); }
    .pill { display:inline-block; padding:.16rem .44rem; border:1px solid currentColor; border-radius:2px; font:700 .64rem var(--mono); letter-spacing:.075em; text-transform:uppercase; white-space:nowrap; }
    .pill-cyan { color:#087b8a; background:#eefafb; }
    .pill-watch { color:#91601e; background:#fff7e9; }
    .pill-high { color:#a3433d; background:#fff0ee; }
    .eyebrow { color:var(--cyan-deep); font:700 .67rem var(--mono); letter-spacing:.14em; text-transform:uppercase; }
    .instrument-title { display:flex; align-items:baseline; gap:.7rem; margin:.12rem 0 .1rem; }
    .instrument-title h1 { margin:0!important; }
    .asset-code { color:#7e9099; font:650 .8rem var(--mono); letter-spacing:.05em; }
    .section-head { display:flex; align-items:center; justify-content:space-between; gap:1rem; padding:.35rem 0 .55rem; border-bottom:1px solid var(--line); margin:.55rem 0 .8rem; }
    .section-head h2 { margin:0!important; }
    .small-mono { color:#6d7c84; font:600 .68rem var(--mono); letter-spacing:.03em; }
    .agent-row { display:grid; grid-template-columns:1.1fr 1fr .8fr; gap:.6rem; align-items:center; padding:.62rem .7rem; border-bottom:1px solid #e8eef0; background:#fff; }
    .agent-row:first-child { border-top:1px solid #e8eef0; }
    .agent-name { color:#203641; font-weight:700; font-size:.82rem; }
    .agent-role { color:#7b8c93; font:600 .62rem var(--mono); letter-spacing:.04em; text-transform:uppercase; margin-top:.16rem; }
    .agent-value { color:#203641; font:700 .86rem var(--mono); font-variant-numeric:tabular-nums; }
    .weight-track { height:4px; background:#e9eff1; margin-top:.28rem; overflow:hidden; }
    .weight-fill { height:4px; background:var(--cyan); }
    .callout { border-left:3px solid var(--cyan); background:#eaf4f5; padding:.75rem .9rem; color:#2e4c54; font-size:.82rem; }
    .callout-amber { border-left-color:var(--amber); background:#fff7e9; color:#62481f; }
    .empty-state { padding:1.2rem; border:1px dashed #b7c8ce; background:#f7f9f9; color:#526a75; }
    .footer-note { border-top:1px solid var(--line); margin-top:2rem; padding-top:.8rem; color:#5c7079; font-size:.7rem; }
    @media(max-width:900px) { .block-container{padding:3rem .9rem 2rem;} h1{font-size:1.8rem!important;} .mast{align-items:flex-start;} }
    @media(max-width:600px) { .block-container{padding:3rem .65rem 1.5rem;} .mast{flex-direction:column;} .instrument-title{flex-wrap:wrap;} [data-testid="stHorizontalBlock"]{gap:.45rem;} [data-testid="stMetricValue"]{font-size:1.1rem;} }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def read_json(path: str, modified: float) -> Any:
    del modified
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data(show_spinner=False, max_entries=10)
def read_prediction_slice(
    path: str,
    modified: float,
    fold: Any,
    method: Any,
    delay: Any,
    budget: Any,
) -> pd.DataFrame:
    """Read a compact cloud partition or chunk-scan the local study archive."""
    del modified
    archive = Path(path)
    if archive.is_dir():
        manifest = json.loads((archive / "index.json").read_text(encoding="utf-8"))
        frames = []
        for entry in manifest["partitions"]:
            if method is not None and str(entry["method"]) != str(method):
                continue
            part = archive / entry["file"]
            if part.resolve().parent != archive.resolve():
                raise ValueError("Prediction partition must stay inside the archive")
            chunk = pd.read_parquet(part)
            for column, selected in (("fold", fold), ("delay", delay), ("budget", budget)):
                if selected is not None:
                    chunk = chunk.loc[pd.to_numeric(chunk[column], errors="coerce").eq(float(selected))]
            frames.append(chunk)
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if "asset_id" in frame:
            frame["asset_id"] = frame["asset_id"].astype(str)
        return frame
    header = list(pd.read_csv(path, nrows=0).columns)
    aliases = {"asset_id", "unit_id", "engine_id"}
    wanted = {
        "fold", "method", "delay", "budget", "asset_id", "unit_id", "engine_id", "cycle",
        "risk", "anomaly_evidence_score", "warning_level", "alert_open", "regime_id",
        "regime_unfamiliar", "condition_change", "window_length", "missing_fraction",
        "data_quality", "quality_status", "affected_sensors", "pattern", "latency_ms",
        "feedback_count", "weight_0", "weight_1", "weight_2", "agent_0", "agent_1", "agent_2",
        "expert_0", "expert_1", "expert_2",
        "raw_0", "raw_1", "raw_2", "episode_id", "failure_cycle", "outcome",
        "agent_reading", "agent_trend", "agent_relationship", "reading_score", "trend_score",
        "relationship_score", "severity_0", "severity_1", "severity_2",
    }
    usecols = [column for column in header if column in wanted or column in aliases]
    if not usecols:
        return pd.DataFrame()
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=100_000, low_memory=False):
        for alias in ("unit_id", "engine_id"):
            if "asset_id" not in chunk and alias in chunk:
                chunk = chunk.rename(columns={alias: "asset_id"})
        mask = pd.Series(True, index=chunk.index)
        for column, selected in (("fold", fold), ("method", method), ("delay", delay), ("budget", budget)):
            if column not in chunk or selected is None:
                continue
            if pd.api.types.is_number(selected):
                mask &= pd.to_numeric(chunk[column], errors="coerce").eq(float(selected))
            else:
                mask &= chunk[column].astype(str).eq(str(selected))
        chosen = chunk.loc[mask].copy()
        if not chosen.empty:
            chunks.append(chosen)
    frame = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=usecols)
    if "asset_id" in frame:
        frame["asset_id"] = frame["asset_id"].astype(str)
    if "cycle" in frame:
        frame["cycle"] = pd.to_numeric(frame["cycle"], errors="coerce")
    return frame


@st.cache_data(show_spinner=False, max_entries=48)
def read_sensor_asset(path: str, asset_id: str, modified: float) -> pd.DataFrame:
    """Read only one engine's raw sensor history, keeping the archive bounded."""
    del modified
    header = list(pd.read_csv(path, nrows=0).columns)
    asset_column = next((name for name in ("asset_id", "unit_id", "engine_id") if name in header), None)
    if asset_column is None:
        return pd.DataFrame()
    selected_columns = [name for name in header if name == asset_column or name == "cycle" or name.startswith("sensor_") or name.startswith("sensor")]
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=selected_columns, chunksize=50_000, low_memory=False):
        match = chunk[asset_column].astype(str).eq(str(asset_id))
        if match.any():
            chunks.append(chunk.loc[match].copy())
    frame = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=selected_columns)
    if asset_column != "asset_id":
        frame = frame.rename(columns={asset_column: "asset_id"})
    if "cycle" in frame:
        frame["cycle"] = pd.to_numeric(frame["cycle"], errors="coerce")
        frame = frame.sort_values("cycle")
    return frame


def as_number(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def fmt(value: Any, digits: int = 3, suffix: str = "") -> str:
    number = as_number(value)
    return f"{number:.{digits}f}{suffix}" if number is not None else "Not recorded"


def first_value(row: pd.Series, names: tuple[str, ...]) -> Any:
    for name in names:
        if name in row.index and pd.notna(row[name]):
            return row[name]
    return None


def severity_text(row: pd.Series) -> tuple[str, str]:
    raw = first_value(row, ("warning_level", "severity", "alert_state"))
    if raw is not None and str(raw).strip():
        text = str(raw).replace("_", " ").strip().title()
        lower = text.lower()
        tone = "high" if any(term in lower for term in ("critical", "high", "alarm")) else "watch" if any(term in lower for term in ("watch", "caution", "elevated")) else "cyan"
        return text, tone
    opened = first_value(row, ("alert_open",))
    if opened is not None and str(opened).lower() in {"true", "1", "yes"}:
        return "Research alert", "watch"
    return "Score only · no saved class", "cyan"


def pill(label: str, tone: str = "cyan") -> str:
    return f'<span class="pill pill-{tone}">{escape(str(label))}</span>'


def safe_text(value: Any, default: str = "Not recorded") -> str:
    if value is None:
        return default
    try:
        if bool(pd.isna(value)):
            return default
    except (TypeError, ValueError):
        pass
    if str(value).strip() == "":
        return default
    return str(value)


def active_record(frame: pd.DataFrame, asset: str, cycle: int) -> pd.Series | None:
    if frame.empty or "asset_id" not in frame or "cycle" not in frame:
        return None
    prefix = frame.loc[(frame["asset_id"].astype(str) == str(asset)) & (frame["cycle"] <= cycle)]
    if prefix.empty:
        return None
    return prefix.sort_values("cycle").iloc[-1]


def render_agent_panel(record: pd.Series) -> None:
    st.markdown('<div class="section-head"><h2>Three-agent reading</h2><span class="small-mono">SCORES + SAVED FUSION WEIGHTS</span></div>', unsafe_allow_html=True)
    agents = [
        ("Reading", "agent_0", "raw_0", "weight_0", "sensor state"),
        ("Trend", "agent_1", "raw_1", "weight_1", "trajectory change"),
        ("Relationships", "agent_2", "raw_2", "weight_2", "cross-sensor context"),
    ]
    html = []
    for label, score_key, raw_key, weight_key, role in agents:
        score_key = next((candidate for candidate in (score_key, f"severity_{score_key[-1]}", f"agent_{label.lower()}", f"{label.lower()}_score") if candidate in record.index), score_key)
        score = fmt(record.get(score_key), 3)
        raw = fmt(record.get(raw_key), 3)
        weight = as_number(record.get(weight_key))
        display_weight = fmt(weight, 3) if weight is not None else "Not recorded"
        width = max(0, min(100, weight * 100)) if weight is not None else 0
        html.append(
            '<div class="agent-row">'
            f'<div><div class="agent-name">{label}</div><div class="agent-role">{role}</div></div>'
            f'<div><div class="agent-value">{score}</div><div class="agent-role">agent score</div></div>'
            f'<div><div class="agent-value">{display_weight}</div><div class="weight-track"><div class="weight-fill" style="width:{width:.1f}%"></div></div><div class="agent-role">weight · raw {raw}</div></div>'
            '</div>'
        )
    st.markdown("".join(html), unsafe_allow_html=True)
    st.caption("Agent scores and fusion weights are shown from the saved study output. The relationship agent flags patterns; it does not establish a physical root cause.")


def render_agent_comparison(record: pd.Series, engine: str, cycle: int) -> None:
    import altair as alt

    methods = [("best_single", "Best single"), ("equal", "Equal"), ("static", "Static")]
    histories = {}
    for method, label in methods:
        saved = read_prediction_slice(str(PREDICTIONS), PREDICTIONS.stat().st_mtime, record.get("fold"), method, record.get("delay"), record.get("budget"))
        histories[label] = saved.loc[(saved["asset_id"].astype(str) == engine) & (saved["cycle"] <= cycle)].sort_values("cycle")
    st.markdown('<div class="section-head"><h2>Three-agent comparison</h2><span class="small-mono">WEIGHTED CONTRIBUTIONS · SELECTED PREFIX</span></div>', unsafe_allow_html=True)
    for index, label in enumerate(("Readings", "Trend", "Relationships")):
        panel = st.container()
        with panel:
            st.markdown(f"### {label}")
            traces = []
            for method_label, history in histories.items():
                if history.empty:
                    continue
                values = history[["cycle", f"expert_{index}", f"weight_{index}"]].copy()
                values["Contribution"] = values[f"expert_{index}"] * values[f"weight_{index}"]
                values["Method"] = method_label
                traces.append(values[["cycle", "Contribution", "Method"]])
            if not traces:
                st.info("No comparison history for this prefix.")
                continue
            chart = alt.Chart(pd.concat(traces, ignore_index=True)).mark_line(strokeWidth=2).encode(
                x=alt.X("cycle:Q", title="Cycle"),
                y=alt.Y("Contribution:Q", title="Weighted contribution", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("Method:N", scale=alt.Scale(domain=["Best single", "Equal", "Static"], range=["#18a9bb", "#ba7528", "#596c99"]), legend=alt.Legend(orient="bottom")),
                strokeDash=alt.StrokeDash("Method:N", legend=None),
                tooltip=["Method:N", "cycle:Q", alt.Tooltip("Contribution:Q", format=".3f")],
            ).properties(height=230)
            st.altair_chart(chart, width="stretch")
            stats = st.columns(3)
            for stat, (_, method_label) in zip(stats, methods):
                with stat:
                    history = histories[method_label]
                    latest = history.iloc[-1] if not history.empty else pd.Series(dtype=object)
                    st.markdown(f"**{method_label}**")
                    st.caption(f"Agent score · {fmt(latest.get(f'expert_{index}'), 3)}")
                    st.caption(f"Weight · {fmt(latest.get(f'weight_{index}'), 3)}")
    st.caption("Agent scores shown here are calibrated inputs used by fusion; each plotted contribution is score × weight. Overlapping lines indicate matching contributions.")


def explain_with_knowledge(record: pd.Series, sensors: pd.DataFrame) -> None:
    st.markdown('<div class="section-head"><h2>Research context</h2><span class="small-mono">EVIDENCE · NOT MAINTENANCE AUTHORITY</span></div>', unsafe_allow_html=True)
    warning = str(first_value(record, ("warning_level", "severity", "alert_state")) or "").strip().casefold()
    if warning not in {"caution", "critical", "watch", "alarm"}:
        st.info("Research retrieval runs only after a Caution or Critical warning has been decided. No passages were requested for this reading.")
        return
    context_key = ":".join(str(record.get(key, "")) for key in ("fold", "method", "asset_id", "cycle"))
    saved = st.session_state.get("engine_knowledge_result")
    if not saved or saved.get("key") != context_key:
        st.caption("Retrieval is a separate, source-backed research step for this saved warning.")
        if not st.button("Retrieve research context", key="retrieve_engine_context", type="primary"):
            return
    try:
        source = ROOT / "src"
        if source.exists() and str(source) not in sys.path:
            sys.path.insert(0, str(source))
        module = importlib.import_module("earlywarning.engine_knowledge")
        explain = getattr(module, "explain_engine_alert")
    except (ImportError, AttributeError):
        st.markdown('<div class="empty-state"><b>Research evidence service is not connected yet.</b><br>Sensor and agent evidence remains available above. No procedure or root cause has been generated.</div>', unsafe_allow_html=True)
        return
    if saved and saved.get("key") == context_key:
        result = saved.get("result")
    else:
        try:
            excluded = {
                "failure_cycle", "outcome", "episode_id", "alert_close", "rul", "remaining_useful_life",
                "target", "label", "eventual_outcome", "retrospective_label",
            }
            operational_record = record.drop(labels=[name for name in excluded if name in record.index]).to_dict()
            result = explain(operational_record, sourceframe=sensors)
        except TypeError:
            try:
                result = explain(operational_record, sensors)
            except Exception as exc:  # surfaced as a transparent service state
                st.warning(f"Research context could not be loaded: {exc}")
                return
        except Exception as exc:
            st.warning(f"Research context could not be loaded: {exc}")
            return
        st.session_state.engine_knowledge_result = {"key": context_key, "result": result}
    if not isinstance(result, dict):
        st.info("The evidence service returned no structured research context.")
        return
    st.markdown(f'<div class="callout">{escape(safe_text(result.get("summary"), "No concise summary was supplied."))}</div>', unsafe_allow_html=True)
    status = safe_text(result.get("support_status", result.get("supportstatus", result.get("status"))), "support status not recorded")
    st.caption(f"Support status: {status}. Retrieved research material is educational context, not approved maintenance guidance.")
    generator = str(result.get("generator_status", ""))
    if "fallback" in generator:
        st.caption("Explanation mode: retrieval-only. The optional local language model is unavailable on this host.")
    passages = result.get("passages", result.get("evidence", [])) or []
    if passages:
        for index, passage in enumerate(passages[:5], start=1):
            if isinstance(passage, dict):
                title = safe_text(passage.get("title"), f"Research passage {index}")
                body = safe_text(passage.get("text", passage.get("content")), "No passage text supplied.")
                citation = safe_text(passage.get("citation"), "Citation not recorded")
                source_name = safe_text(passage.get("source"), "Source not recorded")
                page = passage.get("page")
                year = passage.get("year")
                url = passage.get("source_url")
            else:
                title, body, citation, source_name, page, year, url = f"Research passage {index}", str(passage), "Citation not recorded", "Source not recorded", None, None, None
            with st.expander(title):
                st.write(body)
                st.caption(" · ".join(part for part in (citation, source_name, f"p. {page}" if page else None, str(year) if year else None) if part))
                if url:
                    st.markdown(f"[Open source]({url})")
    else:
        st.info("No matching passages were returned for this alert.")
    supported = result.get("document_supported_checks", []) or []
    if supported:
        st.markdown("**Quoted research excerpts selected for this context**")
        st.caption("These are verbatim research sentences, not instructions or maintenance checks.")
        for item in supported[:3]:
            if isinstance(item, dict):
                st.markdown(f"> {safe_text(item.get('text'))}")
                citation = safe_text(item.get("citation"), "Citation not recorded")
                page = item.get("page")
                st.caption(citation + (f" · p. {page}" if page else ""))
    uncertainty = result.get("uncertainty", []) or []
    if uncertainty:
        with st.expander("Evidence limits", expanded=False):
            for item in uncertainty:
                st.write(f"- {item}")


def render_sidebar(run_meta: dict[str, Any], split_rows: list[dict[str, Any]]) -> tuple[str, str, Any, str, Any, Any, str]:
    st.sidebar.markdown("<div class='brandline'><span class='brandmark'><span>✳</span></span> ENGINE / MONITOR</div>", unsafe_allow_html=True)
    st.sidebar.caption("FD002 · OFFLINE STUDY CONSOLE")
    st.sidebar.markdown("---")
    page = st.sidebar.radio("Workspace", ["Fleet", "Engine monitor", "Study & method"], label_visibility="visible")
    st.sidebar.markdown("---")
    asset_fold: dict[str, Any] = {}
    for split in split_rows:
        for asset in split.get("test_assets", split.get("held_out_assets", split.get("heldout_assets", []))):
            asset_fold.setdefault(str(asset), split.get("fold", 0))
    engines = sorted(asset_fold, key=lambda value: int(value) if value.isdigit() else value)
    if not engines:
        engines = ["1"]
        asset_fold["1"] = 0
    selected_default = st.session_state.get("selected_engine")
    if selected_default not in engines:
        selected_default = engines[0]
    engine = st.sidebar.selectbox(
        "Held-out engine · fold mapped",
        engines,
        index=engines.index(selected_default),
        format_func=lambda value: f"ENGINE {int(value):03d}  ·  FOLD {asset_fold[value]}" if value.isdigit() else f"ENGINE {value}  ·  FOLD {asset_fold[value]}",
        key="engine_selector",
    )
    fold = asset_fold[engine]
    method = "proposed"
    delay_values = run_meta.get("delays", [0]) or [0]
    delay = st.sidebar.selectbox("Feedback delay · cycles", delay_values, index=0)
    budgets = run_meta.get("budgets", [5]) or [5]
    budget = st.sidebar.selectbox("False alarm budget · /1,000", budgets, index=0)
    st.sidebar.markdown("---")
    retrospective = st.sidebar.toggle(
        "Retrospective evaluation",
        value=False,
        help="Reveals end-of-trajectory labels and evaluation outcomes. Leave off for prefix replay.",
    )
    st.sidebar.caption("FD002 research trajectories · no live engine connection")
    return page, engine, fold, method, delay, budget, "retrospective" if retrospective else "operational"


def run() -> None:
    st.markdown(
        '<div class="mast"><div class="brandline"><span class="brandmark"><span>✳</span></span> ENGINE / CONDITION MONITOR</div><div class="meta">RESEARCH REPLAY &nbsp;·&nbsp; NASA C-MAPSS / FD002</div></div>',
        unsafe_allow_html=True,
    )
    if not PREDICTIONS.exists() or not SPLITS.exists():
        st.markdown('<div class="eyebrow">FLIGHT DECK / STUDY 02</div>', unsafe_allow_html=True)
        st.title("Engine monitor")
        st.markdown('<div class="research-strip"><span class="live-dot"></span><b>OFFLINE REPLAY</b><span>Waiting for the completed FD002 study in <code>results/engine_fd002</code>.</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="empty-state">The revised monitor reads only the new FD002 study. It does not silently fall back to legacy milling runs.<br><br>Expected artifacts: <code>predictions.csv</code>, <code>sensors.csv</code>, and <code>splits.json</code>.</div>', unsafe_allow_html=True)
        return

    progress_path = RUN_DIR / "progress.json"
    if progress_path.exists():
        try:
            progress = read_json(str(progress_path), progress_path.stat().st_mtime)
            if str(progress.get("status", "")).lower() not in {"complete", "completed", "success"}:
                st.markdown('<div class="eyebrow">FLIGHT DECK / STUDY 02</div>', unsafe_allow_html=True)
                st.title("FD002 study is still building")
                st.markdown('<div class="research-strip"><span class="live-dot"></span><b>OFFLINE REPLAY</b><span>Only completed study artifacts are shown in this monitor.</span></div>', unsafe_allow_html=True)
                st.json(progress)
                return
        except Exception as exc:
            st.warning(f"Run progress could not be verified ({exc}); showing the saved artifacts with a research-only label.")

    try:
        split_payload = read_json(str(SPLITS), SPLITS.stat().st_mtime)
        run_manifest_path = RUN_DIR / "run_manifest.json"
        config_path = RUN_DIR / "config.json"
        manifest = read_json(str(run_manifest_path), run_manifest_path.stat().st_mtime) if run_manifest_path.exists() else {}
        config = read_json(str(config_path), config_path.stat().st_mtime) if config_path.exists() else {}
        split_rows = split_payload if isinstance(split_payload, list) else split_payload.get("folds", [])
    except Exception as exc:
        st.error(f"Could not read the FD002 study manifest: {exc}")
        return

    page, selected_engine, fold, method, delay, budget, replay_mode = render_sidebar(manifest | config, split_rows)
    split = next((row for row in split_rows if str(row.get("fold")) == str(fold)), {})
    held_out = split.get("test_assets", split.get("held_out_assets", split.get("heldout_assets", [])))
    held_out = sorted({str(asset) for asset in held_out}, key=lambda value: int(value) if value.isdigit() else value)
    if not held_out:
        st.error(f"No held-out engines are listed for fold {fold}.")
        return

    try:
        modified = PREDICTIONS.stat().st_mtime
        with st.spinner("Reading the selected saved method and split…"):
            frame = read_prediction_slice(str(PREDICTIONS), modified, None, method, delay, budget)
    except Exception as exc:
        st.error(f"Could not read the saved prediction archive: {exc}")
        return
    if frame.empty:
        st.markdown('<div class="empty-state">This fold and method do not have saved prediction rows yet. Check that the study has completed.</div>', unsafe_allow_html=True)
        return
    if "risk" not in frame and "anomaly_evidence_score" in frame:
        frame["risk"] = frame["anomaly_evidence_score"]
    if "asset_id" not in frame or "cycle" not in frame:
        st.error("The prediction archive needs asset_id and cycle fields for replay.")
        return

    st.session_state.selected_engine = selected_engine

    selected_rows = frame.loc[(frame["asset_id"].astype(str) == selected_engine) & (frame["fold"].astype(str) == str(fold))].sort_values("cycle")
    if selected_rows.empty:
        st.info("No rows found for this held-out engine in the selected method.")
        return
    min_cycle = int(selected_rows["cycle"].min())
    max_cycle = int(selected_rows["cycle"].max())
    start_cycle = min(max_cycle, max(min_cycle, int(min_cycle + (max_cycle - min_cycle) * .35)))
    if st.session_state.get("replay_key_engine") != f"{fold}:{selected_engine}:{method}:{delay}:{budget}":
        st.session_state.replay_cycle = start_cycle
        st.session_state.replay_playing = False
        st.session_state.replay_key_engine = f"{fold}:{selected_engine}:{method}:{delay}:{budget}"
    if "replay_cycle" not in st.session_state:
        st.session_state.replay_cycle = start_cycle

    sensor_frame = pd.DataFrame()
    if SENSORS.exists():
        sensor_frame = read_sensor_asset(str(SENSORS), selected_engine, SENSORS.stat().st_mtime)
    retrospective = replay_mode == "retrospective"

    st.markdown('<div class="research-strip"><span class="live-dot"></span><b>OFFLINE PREFIX REPLAY</b><span>Only observations at or before the selected cycle are shown.</span><span style="margin-left:auto">TRAINING STUDY · NOT A LIVE ENGINE</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="eyebrow">FLEET / HELD-OUT FOLD {fold}</div>', unsafe_allow_html=True)

    if page == "Study & method":
        render_study_page(manifest | config, split_rows, frame, method, delay, budget, sorted({str(asset) for row in split_rows for asset in row.get("test_assets", [])}), retrospective)
        return
    if page == "Fleet":
        all_engines = sorted(frame["asset_id"].astype(str).unique(), key=lambda value: int(value) if value.isdigit() else value)
        render_fleet_page(frame, all_engines, selected_engine, fold, int(frame["cycle"].max()))
    else:
        render_engine_page(frame, selected_rows, sensor_frame, selected_engine, fold, method, min_cycle, max_cycle, retrospective)
    st.markdown('<div class="footer-note">FD002 research monitor · replay is cycle-bounded · anomaly evidence score is not a failure probability · research corpus is not approved maintenance guidance.</div>', unsafe_allow_html=True)


def render_fleet_page(frame: pd.DataFrame, options: list[str], selected_engine: str, fold: Any, selected_max_cycle: int) -> None:
    st.markdown('<div class="instrument-title"><h1>Fleet board</h1><span class="asset-code">FD002 · HELD-OUT ENGINES</span></div>', unsafe_allow_html=True)
    st.caption("Choose an engine to inspect its replay trace. Fleet values use each engine’s latest saved row through its available prefix; this is a research sample, not current live status.")
    cutoff = st.slider("Compare through cycle", min_value=1, max_value=max(1, selected_max_cycle), value=min(max(1, selected_max_cycle), st.session_state.replay_cycle), step=1, key="fleet_cutoff")
    latest_rows = []
    for asset in options:
        prefix = frame.loc[(frame["asset_id"].astype(str) == asset) & (frame["cycle"] <= cutoff)]
        if prefix.empty:
            continue
        latest = prefix.sort_values("cycle").iloc[-1].copy()
        tone_label, tone = severity_text(latest)
        latest_rows.append({
            "Engine": f"{int(asset):03d}" if asset.isdigit() else asset,
            "Fold": latest.get("fold", "Not recorded"),
            "Through cycle": int(latest["cycle"]),
            "Anomaly evidence score": as_number(first_value(latest, ("risk", "anomaly_evidence_score"))),
            "Research band": tone_label,
            "Data quality": safe_text(first_value(latest, ("data_quality", "quality_status")), "Not recorded"),
            "Affected sensors": safe_text(latest.get("affected_sensors"), "Not recorded"),
            "_severity": tone,
        })
    board = pd.DataFrame(latest_rows)
    caution_count = int(board["Research band"].astype(str).str.casefold().eq("caution").sum()) if not board.empty else 0
    critical_count = int(board["Research band"].astype(str).str.casefold().eq("critical").sum()) if not board.empty else 0
    quality_flags = int((~board["Data quality"].astype(str).str.casefold().isin({"available", "observed", "complete", "ok"})).sum()) if not board.empty else 0
    normal_count = int(board["Research band"].astype(str).str.casefold().eq("normal").sum()) if not board.empty else 0
    c1,c2,c3,c4 = st.columns([1.1,1.1,1.1,1.1])
    c1.metric("Held-out engines visible", f"{len(options):03d}")
    c2.metric("Normal", f"{normal_count:03d}")
    c3.metric("Caution", f"{caution_count:03d}")
    c4.metric("Critical", f"{critical_count:03d}")
    st.markdown(f'<div class="callout callout-amber">Saved warning bands summarize this research replay. Anomaly evidence scores are not probabilities and bands are not maintenance limits. Engines with quality flags: {quality_flags:03d}.</div>', unsafe_allow_html=True)
    if board.empty:
        st.info("No engine has a reading through this cycle. Increase the cycle cutoff.")
        return
    board = board.sort_values("Anomaly evidence score", ascending=False, na_position="last")
    display = board.drop(columns="_severity").set_index("Engine")
    st.dataframe(display, width="stretch", height=500, hide_index=False, column_config={"Anomaly evidence score": st.column_config.NumberColumn(format="%.3f"), "Through cycle": st.column_config.NumberColumn(format="%d")})
    selected = board.loc[board["Engine"] == (f"{int(selected_engine):03d}" if selected_engine.isdigit() else selected_engine)]
    if not selected.empty:
        label, tone = severity_text(frame.loc[(frame["asset_id"].astype(str) == selected_engine) & (frame["cycle"] <= cutoff)].sort_values("cycle").iloc[-1])
        st.markdown(f"Selected engine {pill(label, tone)} · cycle {cutoff}", unsafe_allow_html=True)
    st.caption("Use the Engine selector in the sidebar, then open Engine monitor for agent weights and sensor-level history.")


def render_replay(max_cycle: int, min_cycle: int, frame: pd.DataFrame, selected_rows: pd.DataFrame, sensor_frame: pd.DataFrame, selected_engine: str, method: str, retrospective: bool) -> None:
    if st.session_state.replay_playing and st.session_state.replay_cycle < max_cycle:
        st.session_state.replay_cycle += 1
    elif st.session_state.replay_cycle >= max_cycle:
        st.session_state.replay_playing = False
    left, play, step, reset = st.columns([5.5,1.2,1.2,1.2])
    with play:
        if st.button("Ⅱ Pause" if st.session_state.replay_playing else "▶ Play", width="stretch", key="play_toggle"):
            st.session_state.replay_playing = not st.session_state.replay_playing
    with step:
        if st.button("+1 cycle", width="stretch", key="step_cycle"):
            st.session_state.replay_cycle = min(max_cycle, int(st.session_state.replay_cycle) + 1)
    with reset:
        if st.button("↺ Start", width="stretch", key="reset_cycle"):
            st.session_state.replay_cycle = min_cycle
            st.session_state.replay_playing = False
    with left:
        cycle = st.slider("Replay through cycle", min_value=min_cycle, max_value=max_cycle, value=int(st.session_state.replay_cycle), step=1, key="replay_cycle")
    st.caption(f"Cycle {cycle} / {max_cycle} · all condition and sensor details are clipped to this prefix · method {method}")
    current = active_record(selected_rows, selected_engine, cycle)
    if current is None:
        st.info("No observation for this engine exists through the selected cycle.")
        return
    evidence_score = first_value(current, ("risk", "anomaly_evidence_score"))
    level, tone = severity_text(current)
    score_value = as_number(evidence_score)
    prefix_count = int(selected_rows["cycle"].le(cycle).sum())
    latest_quality = safe_text(first_value(current, ("data_quality", "quality_status")), "Not recorded")
    a,b,c,d = st.columns([1.1,1.25,1.25,1.3])
    a.metric("Anomaly evidence score", fmt(evidence_score, 3))
    b.markdown(f"<div style='padding:.1rem .75rem .4rem;border-left:2px solid var(--line)'><div class='meta'>RESEARCH BAND</div><div style='padding-top:.5rem'>{pill(level,tone)}</div></div>", unsafe_allow_html=True)
    c.metric("Prefix observations", f"{prefix_count:03d}")
    d.metric("Data quality", latest_quality)
    if retrospective and any(field in current.index for field in ("failure_cycle", "outcome", "episode_id")):
        st.markdown('<div class="callout callout-amber"><b>RETROSPECTIVE EVALUATION ENABLED.</b> Fields below may include eventual trajectory outcomes and must not be read as information available at the selected cycle.</div>', unsafe_allow_html=True)

    chart_tab, agent_tab, evidence_tab = st.tabs(["Condition trace", "Agent comparison", "Research evidence"])
    with chart_tab:
        st.markdown('<div class="section-head"><h2>Sensor history</h2><span class="small-mono">MEASURED INPUTS · SELECTED PREFIX ONLY</span></div>', unsafe_allow_html=True)
        prefix_sensors = sensor_frame.loc[sensor_frame["cycle"] <= cycle] if not sensor_frame.empty and "cycle" in sensor_frame else pd.DataFrame()
        sensor_cols = [column for column in prefix_sensors.columns if column.startswith("sensor_") or column.startswith("sensor")]
        affected = [name.strip() for name in str(current.get("affected_sensors", "")).split(";") if name.strip()]
        available_affected = [name for name in affected if name in sensor_cols]
        defaults = available_affected[:4] if available_affected else sensor_cols[:3]
        if sensor_cols:
            selected_channels = st.multiselect("Signals", sensor_cols, default=defaults, max_selections=6, help="Affected sensors come from the saved study output where present; otherwise pick channels to inspect.")
            if selected_channels:
                raw_trace = prefix_sensors.set_index("cycle")[selected_channels].apply(pd.to_numeric, errors="coerce")
                scale = st.selectbox("Signal scale", ["Raw recorded values", "Relative change from first reading (%)"], index=1, key="sensor_scale")
                if scale == "Relative change from first reading (%)":
                    baseline = raw_trace.iloc[0]
                    valid_baseline = baseline.abs() > 1e-12
                    trace = (raw_trace.loc[:, valid_baseline] - baseline.loc[valid_baseline]) / baseline.loc[valid_baseline].abs() * 100
                    omitted = list(baseline.index[~valid_baseline])
                    if omitted:
                        st.caption("Relative change is unavailable for zero-baseline channels: " + ", ".join(omitted) + ". Their exact readings remain in the table below.")
                    scale_label = "Relative change from first reading (%)"
                else:
                    trace = raw_trace
                    scale_label = "Recorded sensor value (units not specified)"
                if not trace.empty:
                    st.line_chart(trace, height=310, width="stretch", x_label="Cycle", y_label=scale_label)
                else:
                    st.info("All selected channels have a zero first reading; relative change cannot be calculated for this prefix.")
                st.caption("Recorded values have no physical units specified in the supplied numeric files. The relative view is computed only from the visible prefix and does not alter the raw readings.")
                readings = prefix_sensors.loc[prefix_sensors["cycle"] == prefix_sensors["cycle"].max(), selected_channels]
                if not readings.empty:
                    st.markdown("**Exact recorded values at the selected cycle**")
                    st.dataframe(readings.T.rename(columns={readings.index[0]: "Raw reading"}), width="stretch")
            else:
                st.info("Select one or more channels to draw a trace.")
        else:
            st.info("No sensor archive was saved for the selected engine.")
        left,right = st.columns(2)
        with left:
            st.markdown("**Affected sensors**")
            st.write(", ".join(affected) if affected else "Not recorded in this study output.")
        with right:
            st.markdown("**Observed pattern**")
            st.write(safe_text(current.get("pattern")))
        if retrospective:
            with st.expander("Retrospective trajectory labels", expanded=False):
                available = [field for field in ("failure_cycle", "outcome", "episode_id") if field in current.index]
                st.json({field: safe_text(current.get(field)) for field in available} if available else {"status": "No retrospective labels recorded"})
    with agent_tab:
        render_agent_comparison(current, selected_engine, cycle)
        regime, change, unfamiliar = st.columns(3)
        regime.metric("Operating regime", safe_text(current.get("regime_id")))
        change.metric("Condition change", safe_text(current.get("condition_change")))
        unfamiliar_flag = current.get("regime_unfamiliar")
        if unfamiliar_flag is None or pd.isna(unfamiliar_flag):
            familiarity = "Not recorded"
        else:
            familiarity = "Unfamiliar" if str(unfamiliar_flag).lower() in {"true", "1", "yes"} else "Not flagged"
        unfamiliar.metric("Regime familiarity", familiarity)
        st.caption("Best single, Equal, and Static compare saved fusion methods. All charts stop at the replay cycle.")
    with evidence_tab:
        st.markdown("### Proposed combined result")
        contributions = []
        for index, label in enumerate(("Readings", "Trend", "Relationships")):
            score = as_number(current.get(f"expert_{index}"))
            weight = as_number(current.get(f"weight_{index}"))
            contributions.append({"Agent": label, "Calibrated score": score, "Proposed weight": weight, "Contribution": score * weight if score is not None and weight is not None else None})
        complete = all(row["Contribution"] is not None for row in contributions)
        combined = sum(row["Contribution"] for row in contributions) if complete else None
        st.metric("Proposed calculated value", fmt(combined, 3))
        st.dataframe(pd.DataFrame(contributions), hide_index=True, width="stretch")
        st.caption("Proposed value = sum of calibrated agent score × Proposed weight at this cycle. This anomaly evidence score is not a failure probability.")
        explain_with_knowledge(current, sensor_frame.loc[sensor_frame["cycle"] <= cycle] if not sensor_frame.empty and "cycle" in sensor_frame else pd.DataFrame())


def render_engine_page(frame: pd.DataFrame, selected_rows: pd.DataFrame, sensor_frame: pd.DataFrame, selected_engine: str, fold: Any, method: str, min_cycle: int, max_cycle: int, retrospective: bool) -> None:
    st.markdown(f'<div class="instrument-title"><h1>Engine monitor</h1><span class="asset-code">ENGINE {int(selected_engine):03d} &nbsp; / &nbsp; FOLD {fold}</span></div>', unsafe_allow_html=True)
    st.caption("Selected asset · replay from the recorded study. The monitor does not imply live telemetry.")
    @st.fragment(run_every="1s")
    def monitor_fragment() -> None:
        render_replay(max_cycle, min_cycle, frame, selected_rows, sensor_frame, selected_engine, method, retrospective)
    monitor_fragment()


def render_study_page(metadata: dict[str, Any], split_rows: list[dict[str, Any]], frame: pd.DataFrame, method: str, delay: Any, budget: Any, held_out: list[str], retrospective: bool) -> None:
    st.markdown('<div class="instrument-title"><h1>Study & method</h1><span class="asset-code">FD002 / RESEARCH BUILD</span></div>', unsafe_allow_html=True)
    st.caption("Run provenance, cohort size, and method context for the current selection.")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Held-out engines", f"{len(held_out):03d}")
    c2.metric("Selected prediction rows", f"{len(frame):,}")
    c3.metric("Method", str(method).replace("_", " ").title())
    c4.metric("Feedback delay", f"{delay} cycles")
    st.markdown('<div class="section-head"><h2>Study identity</h2><span class="small-mono">SAVED RUN METADATA</span></div>', unsafe_allow_html=True)
    description = {
        "Dataset": metadata.get("dataset", metadata.get("dataset_name", "NASA C-MAPSS FD002")),
        "Run status": metadata.get("status", metadata.get("run_status", "Completed study artifacts")),
        "Budget": f"{budget} false alarms / 1,000 cycles (configured research value)",
        "Study scope": "Supplied FD002 training trajectories; cross-validation held-out engines",
        "Sensor count": metadata.get("sensor_count", 21),
        "Prediction score": "Anomaly evidence score. It is not a calibrated failure probability.",
        "Threshold provenance": "Use thresholds saved by the study; no operational limit is asserted here.",
    }
    st.dataframe(pd.DataFrame([{"Field": key, "Value": str(value)} for key, value in description.items()]), hide_index=True, width="stretch")
    with st.expander("Fold membership", expanded=False):
        rows = []
        for split in split_rows:
            rows.append({"Fold": split.get("fold"), "Train engines": len(split.get("train_assets", [])), "Held-out engines": len(split.get("test_assets", split.get("held_out_assets", [])))})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.markdown('<div class="section-head"><h2>Retrospective evaluation</h2><span class="small-mono">HELD-OUT OUTCOMES · AGGREGATE VIEW</span></div>', unsafe_allow_html=True)
    if not retrospective:
        st.info("Held-out outcome metrics are hidden during prefix replay. Turn on Retrospective evaluation in the sidebar to review the saved study comparison.")
    else:
        summary_path = RUN_DIR / "summary.json"
        if not summary_path.exists():
            st.info("No retrospective summary has been saved for this run.")
        else:
            summary = read_json(str(summary_path), summary_path.stat().st_mtime)
            results = pd.DataFrame(summary if isinstance(summary, list) else summary.get("rows", []))
            if not results.empty and {"min_lead", "method", "event_recall"}.issubset(results.columns):
                selected = results.loc[
                    (pd.to_numeric(results["min_lead"], errors="coerce") == 5)
                    & (pd.to_numeric(results.get("delay", 0), errors="coerce") == float(delay))
                    & (pd.to_numeric(results.get("budget", 0), errors="coerce") == float(budget))
                ].copy()
                selected["Event recall"] = pd.to_numeric(selected["event_recall"], errors="coerce")
                selected["False alarms / 1,000 cycles"] = pd.to_numeric(selected.get("false_alarms_per_1000"), errors="coerce")
                selected["Failures missed"] = pd.to_numeric(selected.get("missed_failures"), errors="coerce")
                selected["Median lead · cycles"] = pd.to_numeric(selected.get("median_lead"), errors="coerce")
                if not selected.empty:
                    by_method = selected.set_index("method")
                    proposed = by_method.loc["proposed"] if "proposed" in by_method.index else None
                    regime = by_method.loc["regime_delayed"] if "regime_delayed" in by_method.index else None
                    readings = by_method.loc["agent_readings"] if "agent_readings" in by_method.index else None
                    if proposed is not None and regime is not None and readings is not None:
                        st.markdown(
                            '<div class="callout callout-amber"><b>The proposed fusion did not lead event recall in this held-out study.</b> '
                            f"At a 5-cycle minimum lead, proposed recall was {proposed['Event recall']:.1%}; regime-delayed fusion was {regime['Event recall']:.1%}; the readings-only ablation was {readings['Event recall']:.1%}. "
                            f"Proposed false-alarm burden was {proposed['False alarms / 1,000 cycles']:.2f} per 1,000 cycles versus {regime['False alarms / 1,000 cycles']:.2f} for regime-delayed fusion. "
                            'These retrospective results show a trade-off; they do not establish an overall winner or operational readiness.</div>',
                            unsafe_allow_html=True,
                        )
                    st.dataframe(
                        selected[["method", "Event recall", "False alarms / 1,000 cycles", "Failures missed", "Median lead · cycles"]].rename(columns={"method": "Method"}).sort_values("Event recall", ascending=False),
                        hide_index=True,
                        width="stretch",
                        column_config={"Event recall": st.column_config.NumberColumn(format="percent"), "False alarms / 1,000 cycles": st.column_config.NumberColumn(format="%.2f"), "Median lead · cycles": st.column_config.NumberColumn(format="%.1f")},
                    )
                    st.caption("Retrospective only · grouped cross-validation over 260 held-out engines · min lead 5 cycles · configured false-alarm budget shown above. This is a research comparison, not a deployed warning guarantee.")
                else:
                    st.info("No retrospective rows match the selected delay and false-alarm budget.")
            else:
                st.info("The saved summary does not contain the expected event-recall metrics.")
    st.markdown('<div class="callout callout-amber"><b>Research status.</b> The data are simulated engine degradation trajectories. The dashboard is an experiment viewer and does not provide maintenance authority, certification, or approved service procedures.</div>', unsafe_allow_html=True)


run()
