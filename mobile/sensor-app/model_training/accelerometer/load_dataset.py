"""Walk input/ and turn each session's RAW_ACCELEROMETERS.txt into a record."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import CHANNEL_INDICES, INPUT_ROOT, LABEL_KEYWORDS

SESSION_NAME_RE = re.compile(
    r"^\d+-\d+km-(?P<driver>D\d)-(?P<behavior>[A-Z]+\d?)-(?P<road>[A-Z]+)$"
)


@dataclass
class Session:
    driver: str          # "D1".."D6"
    behavior: str        # "DROWSY" | "NORMAL" | "NORMAL1" | "NORMAL2" | ...
    road: str            # "MOTORWAY" | "SECONDARY"
    label: int           # 0 = drowsy, 1 = normal
    session_dir: Path
    signal: np.ndarray   # shape (N, len(CHANNEL_INDICES))


def _parse_session_name(name: str) -> tuple[str, str, str] | None:
    m = SESSION_NAME_RE.match(name)
    if not m:
        return None
    return m.group("driver"), m.group("behavior"), m.group("road")


def _behavior_to_label(behavior: str) -> int | None:
    # Strip trailing digits ("NORMAL1" -> "NORMAL") so multiple normal runs map
    # to the same label.
    base = re.sub(r"\d+$", "", behavior)
    return LABEL_KEYWORDS.get(base)


def load_signal(path: Path) -> np.ndarray:
    # RAW_ACCELEROMETERS.txt is whitespace-separated floats, no header. We pull
    # only the channels we care about and discard the rest.
    arr = np.loadtxt(path, dtype=np.float32)
    return arr[:, list(CHANNEL_INDICES)]


def load_sessions(root: Path = INPUT_ROOT) -> list[Session]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Input root not found: {root}")

    sessions: list[Session] = []
    for driver_dir in sorted(root.iterdir()):
        if not driver_dir.is_dir() or not driver_dir.name.startswith("D"):
            continue
        for session_dir in sorted(driver_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            parsed = _parse_session_name(session_dir.name)
            if not parsed:
                continue
            driver, behavior, road = parsed
            label = _behavior_to_label(behavior)
            if label is None:
                continue
            raw_path = session_dir / "RAW_ACCELEROMETERS.txt"
            if not raw_path.exists():
                continue
            sessions.append(
                Session(
                    driver=driver,
                    behavior=behavior,
                    road=road,
                    label=label,
                    session_dir=session_dir,
                    signal=load_signal(raw_path),
                )
            )
    return sessions


def summarize(sessions: list[Session]) -> None:
    by_driver: dict[str, dict[int, int]] = {}
    for s in sessions:
        by_driver.setdefault(s.driver, {0: 0, 1: 0})[s.label] += 1
    print(f"Loaded {len(sessions)} sessions")
    for driver in sorted(by_driver):
        d, n = by_driver[driver][0], by_driver[driver][1]
        print(f"  {driver}: drowsy={d} normal={n}")


if __name__ == "__main__":
    sessions = load_sessions()
    summarize(sessions)
