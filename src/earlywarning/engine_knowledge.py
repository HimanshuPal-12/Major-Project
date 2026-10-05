"""Source-cited research retrieval for the simulated C-MAPSS FD002 workflow.

This knowledge layer runs only after a warning has already been decided. It
retrieves educational research references; it is not an engine diagnostic, an
approved maintenance procedure, or a source of operational instructions.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = _PROJECT_ROOT / "corpus" / "manifest.json"
DEFAULT_MODEL_PATH = _PROJECT_ROOT / ".models" / "qwen2.5-0.5b"
_WARNINGS = {"caution", "critical", "watch", "alarm"}
_AGENT_NAMES = {
    "agent_0": "readings agent",
    "agent_1": "trend agent",
    "agent_2": "sensor relationship agent",
    "agent_readings": "readings agent",
    "agent_trend": "trend agent",
    "agent_relationship": "sensor relationship agent",
}
_SENSOR_NAME = re.compile(r"^sensor_\d+$", re.I)
_SETTING_NAME = re.compile(r"^setting_\d+$", re.I)
_TOKEN_RE = re.compile(r"[a-z0-9_]+", re.I)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[])|\n{2,}")
_MODEL_CACHE: dict[str, tuple[Any, Any] | Exception] = {}
_MODEL_LOCK = threading.Lock()


def _read_manifest(path: str | Path) -> tuple[list[dict[str, Any]], Path]:
    manifest = Path(path).resolve()
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    rows = raw.get("documents", raw) if isinstance(raw, dict) else raw
    if not isinstance(rows, list):
        raise ValueError("Engine knowledge manifest must be a list or contain 'documents'")
    return rows, manifest.parent


def _content(doc: Mapping[str, Any], base: Path) -> tuple[str | None, str | None]:
    rel = str(doc.get("content_path", ""))
    if not rel:
        return None, "missing_content_path"
    p = (base / rel).resolve()
    try:
        p.relative_to(base.resolve())
        body = p.read_text(encoding="utf-8").strip()
    except (OSError, ValueError, UnicodeError):
        return None, "missing_or_unsafe_content_path"
    actual = hashlib.sha256(body.encode("utf-8")).hexdigest()
    if actual.casefold() != str(doc.get("content_hash", "")).casefold():
        return None, "content_hash_mismatch"
    if not body:
        return None, "empty_content"
    return body, None


def _equipment_matches(doc: Mapping[str, Any], equipment: str) -> bool:
    listed = doc.get("equipment", [])
    if isinstance(listed, str):
        listed = [listed]
    target = equipment.strip().casefold()
    return bool(target) and any(str(item).strip().casefold() == target for item in listed)


def _eligible_documents(rows: list[dict[str, Any]], base: Path, equipment: str):
    if not rows:
        return [], [{"doc_id": "*", "reason": "empty_corpus"}]
    exclusions: list[dict[str, str]] = []
    applicable = [doc for doc in rows if _equipment_matches(doc, equipment)]
    groups: dict[str, set[str]] = {}
    for doc in applicable:
        group = str(doc.get("logical_doc_id") or doc.get("conflict_group") or doc.get("doc_id", ""))
        groups.setdefault(group, set()).add(str(doc.get("revision", "")))
    conflicts = {group for group, revisions in groups.items() if len(revisions) > 1}
    accepted: list[tuple[dict[str, Any], str]] = []
    for doc in rows:
        doc_id = str(doc.get("doc_id", "<missing-id>"))
        if not _equipment_matches(doc, equipment):
            exclusions.append({"doc_id": doc_id, "reason": "wrong_equipment"})
            continue
        group = str(doc.get("logical_doc_id") or doc.get("conflict_group") or doc.get("doc_id", ""))
        if group in conflicts:
            exclusions.append({"doc_id": doc_id, "reason": "conflicting_revisions"})
            continue
        scope = str(doc.get("scope", ""))
        review = str(doc.get("review_status", ""))
        if scope == "site_approved_sop":
            if review != "site_approved":
                exclusions.append({"doc_id": doc_id, "reason": "unverified_sop"})
                continue
            # This academic engine knowledge product intentionally does not
            # surface site SOPs; that requires a separate scoped product review.
            exclusions.append({"doc_id": doc_id, "reason": "operational_sop_not_enabled"})
            continue
        if scope != "research_reference" or review != "source_verified":
            exclusions.append({"doc_id": doc_id, "reason": "not_source_verified_research"})
            continue
        body, reason = _content(doc, base)
        if reason:
            exclusions.append({"doc_id": doc_id, "reason": reason})
            continue
        accepted.append((doc, body or ""))
    if not applicable:
        exclusions.append({"doc_id": "*", "reason": "no_equipment_applicable_sources"})
    if not rows:
        exclusions.append({"doc_id": "*", "reason": "empty_corpus"})
    return accepted, exclusions


def _value(row: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and not (isinstance(value, float) and pd.isna(value)):
            return value
    return default


def _to_float(value: Any) -> float | None:
    try:
        parsed = float(value)
        return parsed if pd.notna(parsed) else None
    except (TypeError, ValueError):
        return None


def _agent_evidence(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = _value(row, "agent_outputs", "agent_observations", "agents", default=None)
    evidence: list[dict[str, Any]] = []
    if isinstance(raw, Mapping):
        for name, value in raw.items():
            if isinstance(value, Mapping):
                score = _to_float(_value(value, "score", "severity", "anomaly_score"))
                evidence.append({"name": _AGENT_NAMES.get(str(name), str(name)), "score": score})
            else:
                evidence.append({"name": _AGENT_NAMES.get(str(name), str(name)), "score": _to_float(value)})
    for col, label in _AGENT_NAMES.items():
        score = _to_float(row.get(col))
        if score is not None and not any(item["name"] == label for item in evidence):
            evidence.append({"name": label, "score": score})
    return evidence


def _measured_evidence(row: Mapping[str, Any], sourceframe=None) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    warning = _value(row, "warning_level", "level", default="")
    asset = _value(row, "asset_id", "engine_id", "unit_id")
    cycle = _to_float(_value(row, "cycle", "operating_cycle"))
    evidence: dict[str, Any] = {
        "warning_level": str(warning),
        "asset_id": str(asset) if asset is not None else None,
        "cycle": int(cycle) if cycle is not None and cycle.is_integer() else cycle,
    }
    for key in ("regime_id", "condition", "operating_condition", "condition_change",
                "regime_unfamiliar", "pattern", "affected_sensors"):
        value = row.get(key)
        if value is not None and not (isinstance(value, float) and pd.isna(value)):
            evidence[key] = value
    evidence["agents"] = _agent_evidence(row)

    sensor_values = {str(k): _to_float(v) for k, v in row.items() if _SENSOR_NAME.match(str(k))}
    sensor_values = {k: v for k, v in sensor_values.items() if v is not None}
    setting_values = {str(k): _to_float(v) for k, v in row.items() if _SETTING_NAME.match(str(k))}
    setting_values = {k: v for k, v in setting_values.items() if v is not None}
    if not sensor_values and sourceframe is not None:
        try:
            frame = pd.DataFrame(sourceframe).copy()
            if cycle is None or asset is None or not {"cycle", "asset_id"}.issubset(frame.columns):
                warnings.append("Sensor context was unavailable because the source frame lacked asset or cycle fields.")
            else:
                frame["_cycle"] = pd.to_numeric(frame["cycle"], errors="coerce")
                # Crucial boundary: same asset and only readings at or before the
                # warning cycle. Future RUL/failure labels are never selected.
                frame = frame.loc[(frame.asset_id.astype(str) == str(asset)) & (frame._cycle <= cycle)]
                frame = frame.sort_values("_cycle", kind="stable")
                if frame.empty:
                    warnings.append("No sensor rows were available for this asset at or before the warning cycle.")
                else:
                    latest = frame.iloc[-1]
                    sensor_values = {str(k): _to_float(latest[k]) for k in frame.columns if _SENSOR_NAME.match(str(k))}
                    setting_values = {str(k): _to_float(latest[k]) for k in frame.columns if _SETTING_NAME.match(str(k))}
                    sensor_values = {k: v for k, v in sensor_values.items() if v is not None}
                    setting_values = {k: v for k, v in setting_values.items() if v is not None}
                    evidence["sensor_context_cycle"] = int(latest["_cycle"])
        except Exception:
            warnings.append("Sensor context could not be read; only fields carried by the alert are shown.")
    if sensor_values:
        evidence["sensor_values"] = sensor_values
    if setting_values:
        evidence["operating_settings"] = setting_values
    return evidence, warnings


def _build_query(evidence: Mapping[str, Any]) -> str:
    """Use alert-local agent outputs and condition metadata, never rerun monitoring."""
    terms: list[str] = ["C-MAPSS FD002 simulated turbofan engine sensor degradation high pressure compressor flow efficiency operating conditions"]
    agents = evidence.get("agents", [])
    if agents:
        terms.extend(str(item["name"]) for item in agents)
    sensors = evidence.get("affected_sensors")
    if isinstance(sensors, str):
        terms.extend(part.strip() for part in re.split(r"[;,|]", sensors) if part.strip())
    elif isinstance(sensors, (list, tuple, set)):
        terms.extend(map(str, sensors))
    for key in ("condition", "operating_condition", "regime_id", "pattern"):
        value = evidence.get(key)
        if value not in (None, ""):
            terms.append(f"{key.replace('_', ' ')} {value}")
    if evidence.get("condition_change"):
        terms.append("operating condition transition")
    if evidence.get("regime_unfamiliar"):
        terms.append("unfamiliar operating condition")
    return " ".join(terms)


def _rank(query: str, docs: list[tuple[dict[str, Any], str]], top_k: int):
    if not query.strip() or not docs or top_k < 1:
        return []
    text = [" ".join((str(doc.get("title", "")), str(doc.get("author", "")), body)) for doc, body in docs]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True,
                                 token_pattern=r"(?u)\b[a-zA-Z0-9_]+\b")
    try:
        matrix = vectorizer.fit_transform([query, *text])
    except ValueError:
        return []
    scores = cosine_similarity(matrix[0:1], matrix[1:]).ravel()
    order = sorted(range(len(docs)), key=lambda i: (-float(scores[i]), str(docs[i][0].get("doc_id", ""))))
    output = []
    for i in order:
        if scores[i] <= 0:
            continue
        doc, body = docs[i]
        output.append((doc, body, float(scores[i])))
        if len(output) >= top_k:
            break
    return output


def _sentence_candidates(passages: list[dict[str, Any]], query: str, limit: int = 8):
    rows: list[dict[str, Any]] = []
    for passage_index, passage in enumerate(passages):
        for raw_sentence in _SENTENCE_RE.split(passage["text"]):
            # Preserve internal whitespace: the displayed quote must remain an
            # exact substring of its cited corpus passage, including PDF lines.
            sentence = raw_sentence.strip(" \t\r\n-•")
            if len(sentence) < 35 or len(sentence) > 500:
                continue
            rows.append({"id": f"S{len(rows) + 1}", "text": sentence,
                         "passage_index": passage_index, "score": passage.get("score", 0.0)})
    if not rows:
        return []
    vectorizer = TfidfVectorizer(stop_words="english", token_pattern=r"(?u)\b[a-zA-Z0-9_]+\b")
    try:
        matrix = vectorizer.fit_transform([query, *[row["text"] for row in rows]])
        scores = cosine_similarity(matrix[0:1], matrix[1:]).ravel()
    except ValueError:
        scores = [0.0] * len(rows)
    for row, score in zip(rows, scores):
        row["relevance"] = float(score)
    rows.sort(key=lambda row: (-row["relevance"], -row["score"], row["id"]))
    selected = rows[:limit]
    for index, row in enumerate(selected, 1):
        row["id"] = f"S{index}"
    return selected


def _load_local_model(model_path: Path):
    key = str(model_path.resolve())
    with _MODEL_LOCK:
        if key in _MODEL_CACHE:
            cached = _MODEL_CACHE[key]
            if isinstance(cached, Exception):
                raise cached
            return cached
        try:
            if not model_path.is_dir():
                raise FileNotFoundError("local_model_not_found")
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            torch.set_num_threads(min(2, max(1, os.cpu_count() or 1)))
            tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True,
                                                      trust_remote_code=False)
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path), local_files_only=True, trust_remote_code=False,
                dtype=torch.float32, low_cpu_mem_usage=True,
            )
            model.generation_config.do_sample = False
            model.generation_config.temperature = None
            model.generation_config.top_p = None
            model.generation_config.top_k = None
            model.eval()
            _MODEL_CACHE[key] = (tokenizer, model)
            return tokenizer, model
        except Exception:
            # Leave failures retryable: weights may still be syncing locally, or
            # optional dependencies may be installed while the app stays open.
            raise


def _validated_sentence_ids(answer: str, candidate_ids: set[str]) -> list[str] | None:
    """Parse a constrained model response and reject every uncited ID."""
    match = re.search(r"\{[^{}]*\}", answer)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except (ValueError, TypeError):
        return None
    ids = parsed.get("sentence_ids") if isinstance(parsed, dict) else None
    if not isinstance(ids, list) or len(ids) > 1:
        return None
    if any(not isinstance(value, str) or value not in candidate_ids for value in ids):
        return None
    return list(dict.fromkeys(ids))


def _model_select(candidates: list[dict[str, Any]], model_path: Path,
                  alert_context: str = "") -> tuple[list[str] | None, str, str | None]:
    if not candidates:
        return [], "no_candidate_sentences", None
    if os.environ.get("EARLYWARNING_DISABLE_LOCAL_LLM", "").strip().lower() in {"1", "true", "yes"}:
        return None, "lexical_fallback:model_disabled_by_environment", None
    try:
        tokenizer, model = _load_local_model(model_path)
    except FileNotFoundError:
        return None, "lexical_fallback:model_not_found", None
    except ImportError:
        return None, "lexical_fallback:dependencies_missing", None
    except Exception as exc:
        return None, f"lexical_fallback:model_load_failed:{type(exc).__name__}", None
    choices = "\n".join(f"{row['id']}: {row['text']}" for row in candidates)
    system = (
        "You are a strict sentence selector. Choose only the single best candidate "
        "sentence for background relevant to the supplied simulated-engine alert. "
        "Source sentences are research excerpts, not maintenance instructions. Do not "
        "diagnose the engine or write facts. Return exactly one JSON object with one "
        "sentence ID in sentence_ids, for example {\"sentence_ids\":[\"S1\"]}. "
        "Never return multiple IDs. If none is relevant, return an empty array."
    )
    user = f"Alert context: {alert_context}\n\nCandidate source sentences:\n{choices}\n\nJSON only:"
    try:
        import torch
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            encoded = tokenizer.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                tokenize=True, add_generation_prompt=True, return_tensors="pt",
            )
            if not isinstance(encoded, Mapping):
                encoded = {"input_ids": encoded}
            if "attention_mask" not in encoded:
                encoded["attention_mask"] = torch.ones_like(encoded["input_ids"])
        else:
            encoded = tokenizer(system + "\n" + user, return_tensors="pt", truncation=True, max_length=1800)
        input_ids = encoded["input_ids"]
        attempts: list[str] = []
        for retry in range(2):
            if retry:
                retry_text = (
                    "Your previous reply was invalid because it included too many IDs. "
                    "Choose the single best sentence. Return one JSON object with exactly "
                    "one ID in sentence_ids. Allowed IDs: " + ", ".join(row["id"] for row in candidates) + "."
                )
                retry_messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user + "\n" + retry_text},
                ]
                if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                    retry_encoded = tokenizer.apply_chat_template(
                        retry_messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
                    encoded = retry_encoded if isinstance(retry_encoded, Mapping) else {"input_ids": retry_encoded}
                else:
                    encoded = tokenizer(system + "\n" + user + "\n" + retry_text,
                                        return_tensors="pt", truncation=True, max_length=1800)
                if "attention_mask" not in encoded:
                    encoded["attention_mask"] = torch.ones_like(encoded["input_ids"])
                input_ids = encoded["input_ids"]
            with torch.inference_mode():
                generated = model.generate(**encoded, do_sample=False, max_new_tokens=24,
                                           pad_token_id=tokenizer.eos_token_id)
            suffix = generated[0][input_ids.shape[1]:]
            answer = tokenizer.decode(suffix, skip_special_tokens=True)
            attempts.append(answer[:300])
            ids = _validated_sentence_ids(answer, {row["id"] for row in candidates})
            if ids is not None:
                return ids, "qwen_local_sentence_selection", json.dumps({"attempts": attempts})
        return None, "lexical_fallback:invalid_model_sentence_ids", json.dumps({"attempts": attempts})
    except Exception as exc:
        return None, f"lexical_fallback:generation_failed:{type(exc).__name__}", None


def _citation(doc: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "doc_id": doc.get("doc_id"), "title": doc.get("title"),
        "author": doc.get("author"), "year": doc.get("year"),
        "source": doc.get("source_path"), "page": doc.get("page"),
        "source_sha256": doc.get("source_sha256"), "source_url": doc.get("source_url"),
        "citation": doc.get("citation"), "scope": doc.get("scope"),
        "revision": doc.get("revision"),
    }


def explain_engine_alert(row: Mapping[str, Any] | pd.Series, sourceframe=None,
                         manifest_path: str | Path = DEFAULT_MANIFEST,
                         top_k: int = 3, model_path: str | Path | None = None) -> dict[str, Any]:
    """Retrieve and explain research context for an already-decided engine alert.

    ``sourceframe`` is optional sensor context. If supplied, only the current
    asset's rows through the alert cycle are considered; target/outcome columns
    are never copied into the evidence bundle. Generation is local-only and
    returns source sentence IDs. The final text always copies exact sentences.
    """
    alert = dict(row)
    evidence, uncertainty = _measured_evidence(alert, sourceframe)
    warning = str(evidence.get("warning_level", "")).strip().casefold()
    base = {
        "summary": "", "measured_evidence": evidence,
        "interpretation": "Research context describes the C-MAPSS simulation and does not diagnose this alert.",
        "document_supported_checks": [], "passages": [], "support_status": "not_retrieved",
        "status": "not_warning", "generator_status": "not_run",
        "uncertainty": uncertainty,
    }
    if warning not in _WARNINGS:
        base["summary"] = "Knowledge retrieval is available only after a Caution or Critical warning has been decided."
        base["uncertainty"].append("No passages were retrieved because no important warning was present.")
        return base
    asset = evidence.get("asset_id") or "unidentified engine"
    cycle = evidence.get("cycle")
    pattern = evidence.get("pattern") or "reported sensor pattern"
    base["summary"] = f"{warning.title()} warning for {asset}" + (f" at cycle {cycle}" if cycle is not None else "") + f": {pattern}."

    try:
        rows, root = _read_manifest(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        base.update(status="retrieval_error", support_status="manifest_unavailable",
                    generator_status="not_run")
        base["uncertainty"].append(f"Knowledge manifest could not be loaded ({type(exc).__name__}).")
        return base
    equipment = "C-MAPSS FD002 simulation"
    eligible, exclusions = _eligible_documents(rows, root, equipment)
    base["exclusions"] = exclusions
    query = _build_query(evidence)
    base["query"] = query
    ranked = _rank(query, eligible, max(0, int(top_k)))
    passages: list[dict[str, Any]] = []
    for doc, body, score in ranked:
        cite = _citation(doc)
        passages.append({"text": body, "score": score, **cite})
    base["passages"] = passages
    if not passages:
        reasons = {item["reason"] for item in exclusions}
        if "conflicting_revisions" in reasons:
            status = "conflicting_revisions"
        elif "unverified_sop" in reasons:
            status = "unverified_sop"
        elif "empty_corpus" in reasons:
            status = "missing_evidence"
        elif "wrong_equipment" in reasons and not eligible:
            status = "wrong_equipment"
        elif any(reason in reasons for reason in ("content_hash_mismatch", "missing_or_unsafe_content_path")):
            status = "source_integrity_failure"
        else:
            status = "no_relevant_research"
        base.update(status="abstained", support_status=status, generator_status="not_run")
        base["uncertainty"].append("No source-verified, equipment-matched research passage supported this query.")
        return base

    candidates = _sentence_candidates(passages, query)
    model_context = {
        "warning_level": evidence.get("warning_level"),
        "agent_observations": evidence.get("agents", []),
        "affected_sensors": evidence.get("affected_sensors"),
        "condition": evidence.get("condition", evidence.get("operating_condition", evidence.get("regime_id"))),
        "condition_change": evidence.get("condition_change"),
        "unfamiliar_condition": evidence.get("regime_unfamiliar"),
        "reported_pattern": evidence.get("pattern"),
    }
    selected_ids, generator_status, selection_trace = _model_select(
        candidates, Path(model_path).expanduser() if model_path else DEFAULT_MODEL_PATH,
        alert_context=json.dumps(model_context, ensure_ascii=True, separators=(",", ":")),
    )
    base["generator_trace"] = {"selector_output": selection_trace, "output_is_untrusted_selection_data": True}
    if selected_ids is None:
        selected_ids = [row["id"] for row in candidates[:3]]
    if not selected_ids:
        base.update(status="abstained", support_status="no_selected_claim",
                    generator_status=generator_status)
        base["uncertainty"].append("The source selector found no sentence it could safely connect to the alert context.")
        return base
    by_id = {row["id"]: row for row in candidates}
    supported: list[dict[str, Any]] = []
    for sentence_id in selected_ids:
        item = by_id[sentence_id]
        citation = {k: v for k, v in passages[item["passage_index"]].items()
                    if k not in {"text", "score"}}
        supported.append({"text": item["text"], "sentence_id": sentence_id, **citation})
    base["document_supported_checks"] = supported
    base["generator_status"] = generator_status
    base["status"] = "evidence_available"
    base["support_status"] = "research_context_available"
    base["uncertainty"].append("Retrieved excerpts are research background about simulated data, not site-approved procedures or proof of an engine fault.")
    return base

