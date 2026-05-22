from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, List

DEFAULT_SCENARIO_DIR = Path("scenarios") / "scenario_1"
DEFAULT_SCENARIO_PATTERN = "scenario_*"
TOPOLOGY_FILENAME = "topology.json"

_SCENARIO_NUM = re.compile(r"(\d+)")


@dataclass(frozen=True)
class ScenarioPaths:
    dir: Path
    csv: Path
    topology: Path
    name: str


def _pick_csv_file(dir_path: Path) -> Path:
    csvs = sorted(dir_path.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in scenario directory '{dir_path}'.")
    if len(csvs) == 1:
        return csvs[0]

    preferred = dir_path / f"{dir_path.name}.csv"
    if preferred.exists():
        return preferred

    matching = [p for p in csvs if p.stem == dir_path.name]
    if matching:
        return matching[0]

    names = ", ".join(p.name for p in csvs)
    raise ValueError(
        f"Multiple CSV files found in '{dir_path}'. Expected exactly one or a file named "
        f"'{dir_path.name}.csv'. Found: {names}"
    )


def resolve_scenario_dir(path: str | Path) -> ScenarioPaths:
    p = Path(path)
    if p.is_file() and p.suffix.lower() == ".csv":
        dir_path = p.parent
    else:
        dir_path = p

    if not dir_path.exists() or not dir_path.is_dir():
        raise FileNotFoundError(f"Scenario directory '{dir_path}' not found.")

    csv_path = _pick_csv_file(dir_path)
    topo_path = dir_path / TOPOLOGY_FILENAME
    if not topo_path.exists():
        raise FileNotFoundError(
            f"Scenario directory '{dir_path}' is missing '{TOPOLOGY_FILENAME}'."
        )

    return ScenarioPaths(dir=dir_path, csv=csv_path, topology=topo_path, name=dir_path.name)


def scenario_index(name: str) -> int | None:
    match = _SCENARIO_NUM.search(name)
    if match:
        return int(match.group(1))
    return None


def sort_scenarios(items: Iterable[ScenarioPaths]) -> List[ScenarioPaths]:
    def key(sp: ScenarioPaths) -> tuple[int | float, str]:
        idx = scenario_index(sp.name)
        return (idx if idx is not None else float("inf"), sp.name)

    return sorted(list(items), key=key)


def discover_scenarios(
    directory: str | Path,
    pattern: str = DEFAULT_SCENARIO_PATTERN,
) -> List[ScenarioPaths]:
    base = Path(directory)
    if not base.exists():
        raise FileNotFoundError(f"Scenario directory '{base}' not found.")
    candidates = [p for p in base.glob(pattern) if p.is_dir()]
    scenarios = [resolve_scenario_dir(p) for p in candidates]
    return sort_scenarios(scenarios)
