from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from .errors import BudgetExceeded


@dataclass
class CallContext:
    deadline: float
    max_http_attempts: int = 24
    max_decision_calls: int = 12
    max_embedding_calls: int = 4
    max_generation_calls: int = 4
    max_semantic_proposer_calls: int = 2
    max_surface_proposer_calls: int = 2
    max_schema_repairs: int = 0
    schema_repairs: int = 0
    repairs: list[dict] = field(default_factory=list)
    max_cost_usd: float | None = None
    http_attempts: int = 0
    decision_calls: int = 0
    embedding_calls: int = 0
    generation_calls: int = 0
    generation_role_calls: dict[str, int] = field(default_factory=dict)
    generations: list[dict] = field(default_factory=list)
    reported_cost_usd: float = 0.0
    unknown_cost_calls: int = 0
    attempts: list[dict] = field(default_factory=list)
    fallbacks: list[dict] = field(default_factory=list)
    # The time source of the deadline. A replay passes a frozen clock, so no wall-clock budget
    # branch is re-decided from the replay's own (much shorter) timing.
    clock: Callable[[], float] = field(default=time.monotonic, repr=False, compare=False)

    @classmethod
    def with_timeout(cls, seconds: float = 60, **kwargs) -> CallContext:
        clock = kwargs.get("clock", time.monotonic)
        return cls(deadline=clock() + seconds, **kwargs)

    def remaining(self) -> float:
        remaining = self.deadline - self.clock()
        if remaining <= 0:
            raise BudgetExceeded("deadline_exceeded")
        return remaining

    def begin_call(self, kind: str) -> None:
        self.remaining()
        if self.max_cost_usd is not None and self.reported_cost_usd >= self.max_cost_usd:
            raise BudgetExceeded("reported_cost_soft_cap_exceeded")
        if kind == "decision":
            if self.decision_calls >= self.max_decision_calls:
                raise BudgetExceeded("decision_call_budget_exceeded")
            self.decision_calls += 1
        elif kind == "embedding":
            if self.embedding_calls >= self.max_embedding_calls:
                raise BudgetExceeded("embedding_call_budget_exceeded")
            self.embedding_calls += 1
        elif kind in {"generation:semantic", "generation:surface", "generation:direct"}:
            role = kind.split(":")[1]
            limit = {
                "semantic": self.max_semantic_proposer_calls,
                "surface": self.max_surface_proposer_calls,
                "direct": self.max_generation_calls,
            }[role]
            if (
                self.generation_calls >= self.max_generation_calls
                or self.generation_role_calls.get(role, 0) >= limit
            ):
                raise BudgetExceeded("generation_call_budget_exceeded")
            self.generation_calls += 1
            self.generation_role_calls[role] = self.generation_role_calls.get(role, 0) + 1
        else:
            raise ValueError("unknown_call_kind")

    def begin_attempt(self) -> None:
        self.remaining()
        if self.max_cost_usd is not None and self.reported_cost_usd >= self.max_cost_usd:
            raise BudgetExceeded("reported_cost_soft_cap_exceeded")
        if self.http_attempts >= self.max_http_attempts:
            raise BudgetExceeded("http_attempt_budget_exceeded")
        self.http_attempts += 1

    def record_cost(self, cost: float | None) -> None:
        if cost is None:
            self.unknown_cost_calls += 1
        else:
            self.reported_cost_usd += cost
