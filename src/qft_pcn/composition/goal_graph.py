"""Goal graph: the DAG of sub-goals (spec §4).

A parent QPCN's target goal is the root; sub-goals are children. Built
top-down, consumed bottom-up. goal_id is a content hash of
(goal_prop, dsl_spec) -- this is what makes cycle detection sound and
cross-sibling lemma sharing real.
"""
from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


def _content_hash(goal_prop: str, dsl_spec: dict) -> str:
    """Stable content hash; dict key order does not matter."""
    canonical = json.dumps(
        {"goal_prop": goal_prop, "dsl_spec": dsl_spec},
        sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class SubGoal:
    """A DSL spec plus its expected proposition type -- the unit a child
    QPCN run proves (spec §4.1)."""
    goal_id: str
    dsl_spec: dict
    goal_prop: str
    boundary: dict
    parent_site: int | None


def make_sub_goal(dsl_spec: dict, *, goal_prop: str, boundary: dict,
                  parent_site: int | None) -> SubGoal:
    """Construct a SubGoal with a content-addressed goal_id."""
    return SubGoal(
        goal_id=_content_hash(goal_prop, dsl_spec),
        dsl_spec=dsl_spec,
        goal_prop=goal_prop,
        boundary=dict(boundary),
        parent_site=parent_site,
    )


class Status(enum.Enum):
    PENDING = "pending"                     # created, not yet dispatched
    ACTIVE = "active"                       # a child QPCN run is in flight
    SOLVED = "solved"                       # true ground state returned; integrated
    FAILED = "failed"                       # did not converge; revision exhausted
    CYCLE = "cycle"                         # already an ancestor under proof
    PENDING_REVISION = "pending_revision"   # child failed; awaiting alternative


@dataclass
class Node:
    goal: SubGoal
    status: Status
    children: list["Node"] = field(default_factory=list)
    result: Any = None                      # ChildResult once SOLVED/FAILED
    revision_attempts: int = 0
    parent: "Node | None" = None
    quarantined: bool = False               # spec §6.6

    def add_child(self, child: "Node") -> None:
        child.parent = self
        self.children.append(child)
