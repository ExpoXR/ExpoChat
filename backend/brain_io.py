"""Structured brain I/O: prompt templates, response parsing, typed results.

Centralizes the brain's prompt/response contracts so the orchestrator and plan_graph
work against a single extraction and validation layer. Prompt builders return ready-to-send
strings; parsers tolerate code fences, prose preambles, and malformed output gracefully.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("ollma.brain_io")

# ---------------------------------------------------------------------------
# Typed response structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BrainVerdict:
    """Parsed supervisor verdict from the verification cycle."""
    passed: bool
    verdict: str
    repair_task: str
    scope_expansion: bool

    def to_json(self) -> str:
        return json.dumps(
            {"passed": self.passed, "verdict": self.verdict,
             "repair_task": self.repair_task, "scope_expansion": self.scope_expansion},
            ensure_ascii=False,
        )


# ---------------------------------------------------------------------------
# JSON extraction (shared)
# ---------------------------------------------------------------------------

def extract_json(text: str) -> dict[str, Any]:
    """Extract a JSON object from brain text, tolerating ```json fences and prose.

    Consolidates the extraction logic previously duplicated in plan_graph.parse_task_graph
    and the inline brain_verdict parser.  Raises ValueError if no valid JSON object found.
    """
    if not text or not text.strip():
        raise ValueError("Empty brain response")
    cleaned = text.strip()
    # Strip ```json ... ``` fences
    if cleaned.startswith("```"):
        parts = cleaned.split("```", 2)
        if len(parts) >= 3:
            cleaned = parts[1]
        else:
            cleaned = cleaned.strip("`")
        cleaned = cleaned.removeprefix("json").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Fallback: find the first balanced { ... } span
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"Brain JSON is malformed: {exc}") from exc
    raise ValueError("No JSON object found in brain response")


# ---------------------------------------------------------------------------
# Verdict parsing
# ---------------------------------------------------------------------------

def parse_verdict(text: str) -> BrainVerdict:
    """Parse a brain verdict response into a typed BrainVerdict.

    Requires structured JSON. If the response can't be parsed, the verdict fails
    closed (passed=False): a pass must be an explicit, well-formed decision — never
    inferred by sniffing "passed":true out of prose, which a discussion of the
    field could accidentally trip.
    """
    try:
        value = extract_json(text)
        return BrainVerdict(
            passed=bool(value.get("passed")),
            verdict=str(value.get("verdict") or ""),
            repair_task=str(value.get("repair_task") or ""),
            scope_expansion=bool(value.get("scope_expansion")),
        )
    except (ValueError, AttributeError):
        log.warning("Unparseable brain verdict; failing closed (passed=False)")
        return BrainVerdict(passed=False, verdict=text, repair_task="", scope_expansion=False)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_plan_prompt(task: str, dossier: str) -> str:
    """Build the plan-writing prompt from user task + research dossier."""
    return (
        "You are supervisor brain. Use worker dossier to write decision-complete Markdown implementation plan. "
        "Include architecture, exact behavior, interfaces, failure handling, tests, and acceptance criteria. "
        "Do not implement or edit files.\n\nUSER TASK:\n" + task + "\n\nWORKER DOSSIER:\n" + dossier
    )


def build_decompose_prompt(plan: str) -> str:
    """Build the task-graph decomposition prompt."""
    return (
        "You are the supervisor brain. Decompose the implementation plan below into a DAG of "
        "independent subtasks a pool of coding agents can execute. Return JSON ONLY:\n"
        '{"subtasks":[{"node_id":"kebab-id","title":"...","spec":"decision-complete instructions",'
        '"depends_on":["other-node-id"],"file_globs":["path/glob/**"],"acceptance_criteria":"...",'
        '"role":"implementation|research|verification",'
        '"complexity":"simple|complex",'
        '"suggested_model":"optional-ollama-model-name-or-null"}]}\n'
        "Rules: keep each subtask independently verifiable; give every subtask a disjoint file_globs "
        "scope; if two subtasks must touch the same files, serialize them with depends_on instead of "
        "running them in parallel; use a single subtask if the plan is not decomposable; "
        "set complexity to \"complex\" for subtasks that touch many files, carry real risk, or need "
        "careful review, else \"simple\"; "
        "suggested_model is optional — omit or set null to let the scheduler choose. No prose.\n\nPLAN:\n"
        + plan
    )


def build_verdict_prompt(plan: str, reports: list[str]) -> str:
    """Build the supervisor verdict prompt."""
    return (
        "Act as supervisor. Judge whether implementation satisfies approved plan using verifier reports. "
        'Return JSON only: {"passed":boolean,"verdict":string,"repair_task":string,"scope_expansion":boolean}. '
        "Set scope_expansion true when a required repair exceeds approved scope; otherwise repair_task must remain inside scope.\n\nPLAN:\n"
        + plan + "\n\nREPORTS:\n" + "\n\n---\n\n".join(reports)
    )


def build_verification_prompt(plan: str, implementation_summary: str) -> str:
    """Build the verification prompt sent to the worker verifier."""
    return (
        "Verify approved implementation independently. Inspect staged code and run relevant safe checks. "
        "Start final answer with PASS or FAIL. Cite exact files and test output.\n\nAPPROVED PLAN:\n"
        + plan + "\n\nIMPLEMENTER SUMMARY:\n" + implementation_summary
    )


def build_subtask_verify_prompt(
    title: str,
    criteria: str,
    summary: str,
    changed_files: list[str] | None = None,
    checks: list[dict[str, Any]] | None = None,
) -> str:
    """Build per-subtask acceptance verdict and downstream handoff prompt."""
    return (
        "You are supervisor brain. A subtask just completed. Check whether the implementation satisfies "
        "the acceptance criteria and create a concise handoff for dependent agents. Return JSON ONLY: "
        '{"passed":boolean,"issues":"string","handoff":{"summary":"string","interfaces":["string"],'
        '"files":["string"],"risks":["string"],"next_steps":["string"]}}.\n\n'
        f"SUBTASK: {title}\n\nACCEPTANCE CRITERIA:\n{criteria or 'Completion summary must be internally consistent.'}"
        f"\n\nCHANGED FILES:\n{json.dumps(changed_files or [], ensure_ascii=False)}"
        f"\n\nCHECKS:\n{json.dumps(checks or [], ensure_ascii=False)}"
        f"\n\nIMPLEMENTATION SUMMARY:\n{summary}"
    )


@dataclass(frozen=True)
class SubtaskVerdict:
    passed: bool
    issues: str
    handoff: dict[str, Any] | None = None


def parse_subtask_verdict(text: str | dict[str, Any]) -> SubtaskVerdict:
    """Parse a per-subtask verification response."""
    if isinstance(text, dict):
        text = str(text.get("content") or "")
    try:
        value = extract_json(text)
        handoff = value.get("handoff") if isinstance(value.get("handoff"), dict) else {}
        return SubtaskVerdict(
            passed=bool(value.get("passed")),
            issues=str(value.get("issues") or ""),
            handoff={
                "summary": str(handoff.get("summary") or "")[:4000],
                "interfaces": [str(v)[:500] for v in handoff.get("interfaces", [])[:40]] if isinstance(handoff.get("interfaces"), list) else [],
                "files": [str(v)[:1000] for v in handoff.get("files", [])[:100]] if isinstance(handoff.get("files"), list) else [],
                "risks": [str(v)[:1000] for v in handoff.get("risks", [])[:40]] if isinstance(handoff.get("risks"), list) else [],
                "next_steps": [str(v)[:1000] for v in handoff.get("next_steps", [])[:40]] if isinstance(handoff.get("next_steps"), list) else [],
            },
        )
    except (TypeError, ValueError, AttributeError):
        # Fail closed: an unparseable verdict is a failed subtask, not a pass.
        log.warning("Unparseable subtask verdict; failing closed (passed=False)")
        return SubtaskVerdict(passed=False, issues=text[:500], handoff={})


def build_provisional_plan_prompt(task: str, workspace_inventory: str) -> str:
    return (
        "Ollama workers are temporarily offline. Create a provisional implementation plan from the "
        "request and bounded workspace inventory. Mark uncertain assumptions. The plan will be refined "
        "after an Ollama research agent becomes available. Return decision-complete Markdown only.\n\n"
        f"USER TASK:\n{task}\n\nWORKSPACE INVENTORY:\n{workspace_inventory}"
    )


def build_refine_plan_prompt(task: str, provisional_plan: str, dossier: str) -> str:
    return (
        "Refine the saved provisional plan using completed Ollama workspace research. Resolve assumptions, "
        "correct file and interface details, and return one decision-complete Markdown plan. No commentary.\n\n"
        f"USER TASK:\n{task}\n\nPROVISIONAL PLAN:\n{provisional_plan}\n\nOLLAMA RESEARCH:\n{dossier}"
    )
