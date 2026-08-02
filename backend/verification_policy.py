"""Deterministic verification policy evaluator.

Centralizes the apply-gate decision: a run may NEVER be applied solely because
an LLM says PASS.  The gate requires:

  1. At least one recorded machine-executed check exited 0.
  2. No recorded check exited non-zero after the last workspace edit.
  3. The brain's semantic verdict passed.

Returns a structured decision with reason codes so the orchestrator, events,
and UI can explain exactly why a run was blocked or allowed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from . import db


@dataclass(frozen=True)
class PolicyDecision:
    """Result of evaluate_apply_gate."""
    allowed: bool
    reasons: list[str]
    evidence_passed: bool
    brain_passed: bool
    check_count: int
    pass_count: int
    fail_after_edit_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reasons": self.reasons,
            "evidence_passed": self.evidence_passed,
            "brain_passed": self.brain_passed,
            "check_count": self.check_count,
            "pass_count": self.pass_count,
            "fail_after_edit_count": self.fail_after_edit_count,
        }


def record_check_evidence(
    run_id: str,
    cycle: int,
    command: str,
    args: list[str],
    exit_code: int,
    output: str,
    duration_ms: int,
    workspace_hash: str,
    node_id: str | None = None,
) -> int:
    """Persist one machine-executed check result. Returns the row id."""
    with db.transaction() as conn:
        cursor = conn.execute(
            "insert into check_evidence("
            "run_id,cycle,command,args_json,exit_code,output,duration_ms,"
            "workspace_hash,node_id,created_at"
            ") values(?,?,?,?,?,?,?,?,?,?)",
            (
                run_id,
                cycle,
                command,
                json.dumps(args, ensure_ascii=False),
                exit_code,
                output[:40_000],
                duration_ms,
                workspace_hash,
                node_id,
                db.utcnow(),
            ),
        )
        return int(cursor.lastrowid)


def get_check_evidence(run_id: str, cycle: int | None = None) -> list[dict[str, Any]]:
    """Retrieve check evidence for a run, optionally filtered by cycle."""
    if cycle is not None:
        return db.all_rows(
            "select * from check_evidence where run_id=? and cycle=? order by id",
            (run_id, cycle),
        )
    return db.all_rows(
        "select * from check_evidence where run_id=? order by id",
        (run_id,),
    )


def evaluate_apply_gate(run_id: str, brain_passed: bool) -> PolicyDecision:
    """Evaluate whether a run's verified stage may be applied to source.

    Policy:
      evidence_passed = (pass_count >= 1) AND (fail_after_edit_count == 0)
      allowed = evidence_passed AND brain_passed

    Only checks run against the *current* post-edit workspace state count. Evidence
    accumulates across repair and scope-expansion re-runs (whose cycle numbers reset
    and overlap), so a stale failure from an earlier attempt must never wedge a run
    that has since been fixed and re-verified. The most recent qualifying pass owns
    the highest-id rows, and every check in a pass shares that stage's workspace_hash;
    restricting the gate to that hash is exactly "no check failed after the last edit".
    """
    evidence = get_check_evidence(run_id)
    if evidence:
        latest = max(evidence, key=lambda e: e["id"])
        latest_hash = latest.get("workspace_hash") or ""
        evidence = (
            [e for e in evidence if (e.get("workspace_hash") or "") == latest_hash]
            if latest_hash
            else [latest]
        )
    pass_count = sum(1 for e in evidence if e["exit_code"] == 0)
    fail_count = sum(1 for e in evidence if e["exit_code"] != 0)

    reasons: list[str] = []

    if not evidence:
        reasons.append("no_check_evidence")
    elif pass_count == 0:
        reasons.append("no_passing_check")

    if fail_count > 0:
        reasons.append("check_failures_present")

    evidence_passed = pass_count >= 1 and fail_count == 0

    if not brain_passed:
        reasons.append("brain_verdict_failed")

    allowed = evidence_passed and brain_passed

    if allowed:
        reasons = ["all_gates_passed"]

    return PolicyDecision(
        allowed=allowed,
        reasons=reasons,
        evidence_passed=evidence_passed,
        brain_passed=brain_passed,
        check_count=len(evidence),
        pass_count=pass_count,
        fail_after_edit_count=fail_count,
    )
