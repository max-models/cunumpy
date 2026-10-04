"""Reusable completion primitives and cross-stream producer dependencies."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

import cunumpy as xp
from cunumpy import _streams


def test_host_stream_and_event_can_be_reused_and_nested():
    with xp.use_backend("numpy"):
        stream = xp.cuda.create_stream()
        event = xp.cuda.create_event()
        for _ in range(3):
            with xp.cuda.stream(stream) as selected, stream:
                assert selected is stream
                assert xp.cuda.record_event(event, stream=stream) is event
                stream.wait_event(event)
                xp.cuda.wait_event(event)
                assert stream.record().done
            assert stream.done and event.done
            event.synchronize()
            stream.synchronize()
        with xp.cuda.stream() as legacy:
            assert legacy is None


def test_cuda_factories_and_event_ordering(monkeypatch):
    calls = []

    class Stream:
        def __init__(self, **kwargs):
            calls.append(("stream", kwargs))

        def wait_event(self, event):
            calls.append(("wait", event))

    class Event:
        def __init__(self, **kwargs):
            calls.append(("event", kwargs))

        def record(self, stream=None):
            calls.append(("record", stream))

    current = Stream()
    monkeypatch.setattr(_streams, "get_backend", lambda: "cupy")
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(
            cuda=SimpleNamespace(
                Stream=Stream, Event=Event, get_current_stream=lambda: current
            )
        ),
    )
    stream = xp.cuda.create_stream(non_blocking=False)
    event = xp.cuda.create_event()
    assert xp.cuda.record_event(event, stream=stream) is event
    xp.cuda.wait_event(event)
    assert calls[-1] == ("wait", event)
    assert ("stream", {"non_blocking": False}) in calls
    assert ("event", {"disable_timing": True}) in calls
    xp.cuda.create_event(timing=True)
    assert calls[-1] == ("event", {"disable_timing": False})
    assert isinstance(xp.cuda.record_event(stream=stream), Event)


@pytest.mark.skipif(not xp.cupy_available(), reason="requires CUDA")
def test_gpu_events_order_reused_producer_and_consumer():
    import cupy as cp

    with xp.use_backend("cupy", strict=True):
        producer, consumer = xp.cuda.create_stream(), xp.cuda.create_stream()
        ready, consumed = xp.cuda.create_event(), xp.cuda.create_event()
        data, result = cp.zeros(1000), cp.zeros(1000)
        cp.cuda.Device().synchronize()
        for step in range(3):
            with producer:
                if step:
                    xp.cuda.wait_event(consumed, stream=producer)
                data.fill(step + 1)
                xp.cuda.record_event(ready, stream=producer)
            with consumer:
                xp.cuda.wait_event(ready, stream=consumer)
                result += data
                xp.cuda.record_event(consumed, stream=consumer)
        consumed.synchronize()
        assert consumed.done
        np.testing.assert_array_equal(cp.asnumpy(result), np.full(1000, 6))
