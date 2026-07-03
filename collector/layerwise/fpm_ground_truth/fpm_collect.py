#!/usr/bin/env python3
"""Collect raw Dynamo/vLLM ForwardPassMetrics rows."""

from __future__ import annotations

import argparse
import csv
import os
import signal
import sys
import tempfile
import time
from pathlib import Path

import zmq
from dynamo.common.forward_pass_metrics import decode
from zmq.utils.monitor import recv_monitor_message

running = True


def stop(_signum, _frame):
    global running
    running = False


def classify_phase(sum_prefill_tokens: int, num_decode_requests: int) -> str:
    if sum_prefill_tokens > 0 and num_decode_requests > 0:
        return "mixed"
    if sum_prefill_tokens > 0:
        return "context"
    if num_decode_requests > 0:
        return "decode"
    return "idle"


def read_segment(segment_file: str | None) -> str:
    if not segment_file:
        return ""
    try:
        return Path(segment_file).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def create_subscriber(endpoint: str):
    """Create a SUB socket whose transport monitor is armed before connect."""
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    # Install the subscription before connect so the first post-handshake FPM
    # message is eligible for delivery; the monitor is likewise armed before
    # connect so a fast local handshake cannot be missed.
    sock.setsockopt(zmq.SUBSCRIBE, b"")
    monitor = sock.get_monitor_socket()
    sock.connect(endpoint)
    sock.setsockopt(zmq.RCVTIMEO, 1000)
    return sock, monitor


def wait_for_transport_ready(
    monitor,
    timeout_seconds: float,
    *,
    clock=time.monotonic,
) -> None:
    """Wait for the ZMTP handshake, not merely the TCP connection event."""
    if timeout_seconds <= 0:
        raise ValueError("transport readiness timeout must be positive")

    handshake_succeeded = getattr(zmq, "EVENT_HANDSHAKE_SUCCEEDED", None)
    if handshake_succeeded is None:
        raise RuntimeError("pyzmq EVENT_HANDSHAKE_SUCCEEDED support is required for fail-closed collector readiness")
    handshake_failures = {
        event
        for event in (
            getattr(zmq, "EVENT_HANDSHAKE_FAILED_NO_DETAIL", None),
            getattr(zmq, "EVENT_HANDSHAKE_FAILED_PROTOCOL", None),
            getattr(zmq, "EVENT_HANDSHAKE_FAILED_AUTH", None),
        )
        if event is not None
    }
    monitor_stopped = getattr(zmq, "EVENT_MONITOR_STOPPED", None)
    deadline = clock() + timeout_seconds

    while True:
        remaining = deadline - clock()
        if remaining <= 0:
            raise TimeoutError(f"timed out after {timeout_seconds:g}s waiting for ZMQ transport handshake")
        poll_ms = max(1, min(100, int(remaining * 1000)))
        if not monitor.poll(poll_ms, zmq.POLLIN):
            continue
        event = recv_monitor_message(monitor)["event"]
        if event == handshake_succeeded:
            return
        if event in handshake_failures:
            raise RuntimeError(f"ZMQ transport handshake failed (event={event})")
        if monitor_stopped is not None and event == monitor_stopped:
            raise RuntimeError("ZMQ transport monitor stopped before handshake")


def publish_ready_marker(path: str | Path, token: str) -> None:
    """Atomically publish the exact per-run token after transport readiness."""
    if not token or "\n" in token or "\r" in token:
        raise ValueError("readiness token must be one non-empty line")
    ready_path = Path(path)
    ready_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=ready_path.parent,
        prefix=f".{ready_path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            # Docker may create this file as a different host UID. The token is
            # not secret, and the host-side driver must be able to read it.
            os.fchmod(stream.fileno(), 0o644)
            stream.write(f"{token}\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, ready_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def is_zeroed_idle_heartbeat(metrics) -> bool:
    """Return whether a wall_time=0 frame matches the FPM idle contract."""
    scheduled = metrics.scheduled_requests
    queued = metrics.queued_requests
    values = (
        scheduled.num_prefill_requests,
        scheduled.sum_prefill_tokens,
        scheduled.var_prefill_length,
        scheduled.sum_prefill_kv_tokens,
        scheduled.num_decode_requests,
        scheduled.sum_decode_kv_tokens,
        scheduled.var_decode_kv_tokens,
        queued.num_prefill_requests,
        queued.sum_prefill_tokens,
        queued.var_prefill_length,
        queued.num_decode_requests,
        queued.sum_decode_kv_tokens,
        queued.var_decode_kv_tokens,
    )
    return metrics.wall_time == 0 and all(value == 0 for value in values)


def main(argv: list[str] | None = None) -> int:
    global running
    running = True
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--detail-output", required=True)
    parser.add_argument("--segment-file", default=None)
    parser.add_argument("--idle-timeout", type=float, default=0.0)
    parser.add_argument("--ready-file", required=True)
    parser.add_argument("--ready-token", required=True)
    parser.add_argument("--ready-timeout", type=float, required=True)
    parser.add_argument("--data-ready-file", default=None)
    parser.add_argument("--data-ready-token", default=None)
    args = parser.parse_args(argv)
    if args.ready_timeout <= 0:
        parser.error("--ready-timeout must be positive")
    if bool(args.data_ready_file) != bool(args.data_ready_token):
        parser.error("--data-ready-file and --data-ready-token must be provided together")
    if args.data_ready_file:
        if Path(args.data_ready_file) == Path(args.ready_file):
            parser.error("--data-ready-file must differ from --ready-file")
        if args.data_ready_token == args.ready_token:
            parser.error("--data-ready-token must differ from --ready-token")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    sock, monitor = create_subscriber(f"tcp://127.0.0.1:{args.port}")
    try:
        with open(args.output, "w", newline="") as f, open(args.detail_output, "w", newline="") as detail_f:
            writer = csv.writer(f)
            detail_writer = csv.writer(detail_f)
            writer.writerow(["num_context_tokens", "num_decode_tokens", "latency_ms"])
            detail_writer.writerow(
                [
                    "phase",
                    "workload_segment",
                    "counter_id",
                    "worker_id",
                    "dp_rank",
                    "sum_prefill_tokens",
                    "num_prefill_requests",
                    "sum_prefill_kv_tokens",
                    "var_prefill_length",
                    "num_decode_requests",
                    "sum_decode_kv_tokens",
                    "var_decode_kv_tokens",
                    "queued_prefill_requests",
                    "queued_prefill_tokens",
                    "queued_var_prefill_length",
                    "queued_decode_requests",
                    "queued_decode_kv_tokens",
                    "queued_var_decode_kv_tokens",
                    "latency_ms",
                ]
            )
            f.flush()
            detail_f.flush()

            wait_for_transport_ready(monitor, args.ready_timeout)
            monitor.close(linger=0)
            monitor = None
            publish_ready_marker(args.ready_file, args.ready_token)
            last_row = time.monotonic()
            data_path_ready = not bool(args.data_ready_file)
            probe_key: tuple[str, int] | None = None
            probe_counter: int | None = None

            while running:
                try:
                    _topic, _seq, payload = sock.recv_multipart()
                except zmq.Again:
                    if args.idle_timeout > 0 and time.monotonic() - last_row > args.idle_timeout:
                        break
                    continue

                metrics = decode(payload)
                if metrics is None:
                    continue

                if not data_path_ready:
                    key = (str(metrics.worker_id), int(metrics.dp_rank))
                    counter = int(metrics.counter_id)
                    if metrics.wall_time > 0:
                        if probe_key is not None and key != probe_key:
                            raise RuntimeError(
                                f"readiness probe active frame changed worker/dp identity from {probe_key!r} to {key!r}"
                            )
                        if probe_counter is not None and counter <= probe_counter:
                            raise RuntimeError(
                                "readiness probe active frame counter was not monotonic: "
                                f"previous={probe_counter} observed={counter}"
                            )
                        probe_key = key
                        probe_counter = counter
                        continue
                    if not is_zeroed_idle_heartbeat(metrics):
                        raise RuntimeError("readiness probe received a nonzero pseudo-idle heartbeat")
                    if probe_key is None or probe_counter is None:
                        continue
                    if key != probe_key:
                        raise RuntimeError(
                            f"readiness probe idle heartbeat changed worker/dp identity from {probe_key!r} to {key!r}"
                        )
                    if counter <= probe_counter:
                        raise RuntimeError(
                            f"readiness probe idle heartbeat counter was stale: active={probe_counter} idle={counter}"
                        )
                    publish_ready_marker(args.data_ready_file, args.data_ready_token)
                    data_path_ready = True
                    last_row = time.monotonic()
                    continue

                if metrics.wall_time <= 0:
                    continue

                scheduled = metrics.scheduled_requests
                queued = metrics.queued_requests
                latency_ms = f"{metrics.wall_time * 1000.0:.3f}"
                phase = classify_phase(
                    int(scheduled.sum_prefill_tokens),
                    int(scheduled.num_decode_requests),
                )
                workload_segment = read_segment(args.segment_file)
                row = [
                    int(scheduled.sum_prefill_tokens),
                    int(scheduled.num_decode_requests),
                    latency_ms,
                ]
                detail_row = [
                    phase,
                    workload_segment,
                    int(metrics.counter_id),
                    metrics.worker_id,
                    int(metrics.dp_rank),
                    int(scheduled.sum_prefill_tokens),
                    int(scheduled.num_prefill_requests),
                    int(scheduled.sum_prefill_kv_tokens),
                    f"{scheduled.var_prefill_length:.3f}",
                    int(scheduled.num_decode_requests),
                    int(scheduled.sum_decode_kv_tokens),
                    f"{scheduled.var_decode_kv_tokens:.3f}",
                    int(queued.num_prefill_requests),
                    int(queued.sum_prefill_tokens),
                    f"{queued.var_prefill_length:.3f}",
                    int(queued.num_decode_requests),
                    int(queued.sum_decode_kv_tokens),
                    f"{queued.var_decode_kv_tokens:.3f}",
                    latency_ms,
                ]
                writer.writerow(row)
                detail_writer.writerow(detail_row)
                f.flush()
                detail_f.flush()
                print(",".join(map(str, row)), flush=True)
                last_row = time.monotonic()
    finally:
        if monitor is not None:
            monitor.close(linger=0)
        sock.close(linger=0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
