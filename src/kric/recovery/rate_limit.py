"""Transport-only request pacing for low-RPM API accounts."""

from __future__ import annotations

import time
from typing import Any, Callable, Mapping


class RateLimitedPostJson:
    """Ensure request starts are separated without changing request payloads."""

    def __init__(
        self,
        post_json: Callable[[str, Mapping[str, str], Mapping[str, Any], float], Any],
        min_interval_seconds: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if min_interval_seconds < 0:
            raise ValueError("minimum request interval must be non-negative")
        self.post_json = post_json
        self.min_interval_seconds = float(min_interval_seconds)
        self.clock = clock
        self.sleep = sleep
        self.last_request_started: float | None = None

    def __call__(
        self,
        url: str,
        headers: Mapping[str, str],
        body: Mapping[str, Any],
        timeout: float,
    ) -> Any:
        now = self.clock()
        if self.last_request_started is not None:
            remaining = self.min_interval_seconds - (now - self.last_request_started)
            if remaining > 0:
                self.sleep(remaining)
                now = self.clock()
        self.last_request_started = now
        return self.post_json(url, headers, body, timeout)


__all__ = ["RateLimitedPostJson"]
