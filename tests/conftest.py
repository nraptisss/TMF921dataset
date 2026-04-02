from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _force_mock_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INFERENCE_BACKEND", "mock")
    monkeypatch.setenv("LOCAL_DEVICE", "cpu")
