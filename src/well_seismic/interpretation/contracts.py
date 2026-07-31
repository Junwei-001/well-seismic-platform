from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class InterpretationTaskSpec:
    """Business semantics for one downstream interpretation task.

    A task describes the geological problem and its result contract. Models,
    input adapters and runners are registered independently and are associated
    with a task through ``ModelSpec.metadata["prediction_task"]``.
    """

    id: str
    name: str
    short_name: str
    description: str
    outputs: tuple[str, ...]
    required_modalities: tuple[str, ...]
    evaluation_metrics: tuple[str, ...]
    order: int
    contract_version: str = "1.0"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
