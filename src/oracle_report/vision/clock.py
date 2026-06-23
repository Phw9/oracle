from __future__ import annotations

import time
from typing import Protocol


class Clock(Protocol):
    def monotonic(self) -> float:
        ...


class SystemClock:
    def monotonic(self) -> float:
        result = time.monotonic()
        return result
