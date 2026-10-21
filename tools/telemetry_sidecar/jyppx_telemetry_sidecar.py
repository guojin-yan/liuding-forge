#!/usr/bin/env python3
"""Protocol-v1 optional observer using official TensorBoard and OTLP schemas.

The C++ caller owns canonical envelope validation and SHA-256. This process
accepts only its bounded JSON projection, checks immutable replay identity, and
never treats JSON or a visualization backend as durable event evidence.
The runner must supervise this process with bounded pipes and kill it on an
operation deadline/cancellation. No shell, credentials, or output paths appear
in event payloads. External effects remain at least once across process restart.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
from typing import Any, BinaryIO
import urllib.error
import urllib.parse
import urllib.request


MAX_LINE = 16 * 1024 * 1024
MAX_RESPONSE = 128 * 1024
MAX_NODES = 200_000
MAX_DEPTH = 32
MAX_EVENTS = 1024
MAX_POINTS = 4096
MAX_TEXT = 4096
MAX_ID = 256
MAX_SEEN = 100_000
MAX_SEEN_BYTES = 32 * 1024 * 1024
MAX_U64 = (1 << 64) - 1
MAX_I64 = (1 << 63) - 1
ACCEPTED, DUPLICATE, SAMPLED_OUT, UNSUPPORTED = 1, 2, 3, 4
OFFICIAL_VERSIONS = {
    "tensorboard": "2.20.0",
    "opentelemetry-proto": "1.37.0",
    "protobuf": "6.33.5",
}


class ProtocolError(Exception):
    """A bounded input violates the fixed protocol or immutable identity."""


class ExportError(Exception):
    """An external observer failed; the runner may replay immutable input."""


def compact(value: Any) -> str:
    """Encode strict compact UTF-8 JSON without nonstandard NaN literals."""
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      sort_keys=True, separators=(",", ":"))


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ProtocolError("duplicate_key")
        result[key] = value
    return result


def _float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ProtocolError("nonfinite_json")
    return value


def strict_json(raw: bytes, maximum: int = MAX_LINE) -> Any:
    """Bound size and nesting before parsing, then bound the parsed tree."""
    if not raw or len(raw) > maximum:
        raise ProtocolError("line_size")
    try:
        text = raw.decode("utf-8", errors="strict")
        in_string, escaped, depth = False, False, 0
        for char in text:
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char in "[{":
                depth += 1
                if depth > MAX_DEPTH:
                    raise ProtocolError("json_depth")
            elif char in "]}":
                depth -= 1
                if depth < 0:
                    raise ProtocolError("json_depth")
        value = json.loads(text, object_pairs_hook=_pairs, parse_float=_float,
                           parse_constant=lambda _: (_ for _ in ()).throw(
                               ProtocolError("nonfinite_json")))
        pending = [(value, 0)]
        nodes = 0
        while pending:
            item, item_depth = pending.pop()
            nodes += 1
            if nodes > MAX_NODES or item_depth > MAX_DEPTH:
                raise ProtocolError("json_capacity")
            if isinstance(item, dict):
                pending.extend((entry, item_depth + 1) for entry in item.values())
                pending.extend((key, item_depth + 1) for key in item)
            elif isinstance(item, list):
                pending.extend((entry, item_depth + 1) for entry in item)
            elif isinstance(item, str):
                encoded = item.encode("utf-8", errors="strict")
                if len(encoded) > MAX_TEXT or "\0" in item:
                    raise ProtocolError("text_capacity")
        return value
    except (UnicodeError, ValueError, RecursionError) as error:
        raise ProtocolError("invalid_json") from error


def fields(value: Any, names: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != names:
        raise ProtocolError("fields")
    return value


def integer(value: Any, minimum: int = 0, maximum: int = MAX_U64) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ProtocolError("integer")
    return value


def text(value: Any, maximum: int = MAX_TEXT, empty: bool = True) -> str:
    if not isinstance(value, str) or "\0" in value:
        raise ProtocolError("text")
    try:
        size = len(value.encode("utf-8", errors="strict"))
    except UnicodeError as error:
        raise ProtocolError("text") from error
    if size > maximum or (not empty and not value):
        raise ProtocolError("text_capacity")
    return value


def mapping(value: Any, maximum: int) -> dict[str, str]:
    if not isinstance(value, dict) or len(value) > maximum:
        raise ProtocolError("mapping")
    for key, item in value.items():
        text(key, MAX_ID, False)
        text(item)
    return value


def digest(value: Any) -> str:
    value = text(value, 64, False)
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ProtocolError("digest")
    return value


def metric(point: Any) -> dict[str, Any]:
    point = fields(point, {"definition", "step", "status", "value", "reason",
                           "context", "sample_count"})
    definition = fields(point["definition"], {
        "schema_version", "metric_id", "version", "unit", "aggregation",
        "direction", "parameters", "context_keys", "definition_digest"})
    integer(definition["schema_version"], 1, 1)
    text(definition["metric_id"], MAX_ID, False)
    integer(definition["version"], 1, (1 << 32) - 1)
    integer(definition["unit"], 1, 9)
    integer(definition["aggregation"], 1, 6)
    integer(definition["direction"], 1, 3)
    mapping(definition["parameters"], 64)
    digest(definition["definition_digest"])
    context_keys = definition["context_keys"]
    if not isinstance(context_keys, list) or len(context_keys) > 32:
        raise ProtocolError("context_schema")
    for key in context_keys:
        text(key, MAX_ID, False)
    if context_keys != sorted(set(context_keys)):
        raise ProtocolError("context_schema")
    context = mapping(point["context"], 32)
    if set(context) != set(context_keys):
        raise ProtocolError("context_schema")
    step = fields(point["step"], {"kind", "value"})
    integer(step["kind"], 1, 5)
    integer(step["value"])
    status = integer(point["status"], 1, 5)
    value = point["value"]
    if status == 1:
        try:
            valid = type(value) in (int, float) and math.isfinite(float(value))
        except OverflowError:
            valid = False
        if not valid:
            raise ProtocolError("finite_value")
        # Foundation canonical bytes identify both signed zeros. A replay read
        # back from canonical storage must retain the same sidecar fingerprint.
        point["value"] = 0.0 if float(value) == 0 else float(value)
    elif value is not None:
        raise ProtocolError("symbolic_value")
    text(point["reason"])
    if status == 5 and not point["reason"]:
        raise ProtocolError("missing_reason")
    if point["sample_count"] is not None:
        integer(point["sample_count"])
        if status == 1 and point["sample_count"] == 0:
            raise ProtocolError("sample_count")
    return point


def point_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) > MAX_POINTS:
        raise ProtocolError("point_capacity")
    return [metric(point) for point in value]


def envelope(value: Any) -> dict[str, Any]:
    """Validate the projection's complete fixed schema, without generic payloads."""
    value = fields(value, {"schema_version", "identity", "time", "correlation",
                           "reliability", "attributes", "payload"})
    integer(value["schema_version"], 1, 1)
    identity = fields(value["identity"], {"stream_id", "event_id", "idempotency_key", "sequence"})
    for key in ("stream_id", "event_id", "idempotency_key"):
        text(identity[key], MAX_ID, False)
    integer(identity["sequence"], 1)
    time = fields(value["time"], {"wall_time_unix_ns", "monotonic_time_ns", "clock_id"})
    if time["wall_time_unix_ns"] is not None:
        integer(time["wall_time_unix_ns"], -(1 << 63), MAX_I64)
    if time["monotonic_time_ns"] is not None:
        integer(time["monotonic_time_ns"])
    text(time["clock_id"], MAX_ID)
    if bool(time["clock_id"]) != (time["monotonic_time_ns"] is not None):
        raise ProtocolError("clock_domain")
    correlation = fields(value["correlation"], {
        "project_id", "run_id", "attempt_id", "trace_id", "span_id", "parent_span_id"})
    for item in correlation.values():
        text(item, MAX_ID)
    if (correlation["span_id"] or correlation["parent_span_id"]) and not correlation["trace_id"]:
        raise ProtocolError("trace_identity")
    if correlation["attempt_id"] and not correlation["run_id"]:
        raise ProtocolError("attempt_identity")
    if correlation["parent_span_id"] and not correlation["span_id"]:
        raise ProtocolError("span_identity")
    if correlation["span_id"] and correlation["span_id"] == correlation["parent_span_id"]:
        raise ProtocolError("span_parent")
    reliability = integer(value["reliability"], 1, 2)
    mapping(value["attributes"], 64)
    payload = value["payload"]
    if not isinstance(payload, dict):
        raise ProtocolError("payload")
    kind = integer(payload.get("kind"), 1, 6)
    if kind == 1:
        fields(payload, {"kind", "name", "terminal", "metrics", "decisions"})
        text(payload["name"], MAX_ID, False)
        if type(payload["terminal"]) is not bool or (payload["terminal"] and reliability != 1):
            raise ProtocolError("terminal")
        point_list(payload["metrics"])
        decisions = payload["decisions"]
        if not isinstance(decisions, list) or len(decisions) > 1024:
            raise ProtocolError("decision_capacity")
        effective = 0
        for decision in decisions:
            fields(decision, {"actor_id", "action", "reason", "effective"})
            text(decision["actor_id"], MAX_ID, False)
            text(decision["action"], MAX_ID, False)
            text(decision["reason"], MAX_TEXT, False)
            if type(decision["effective"]) is not bool:
                raise ProtocolError("decision")
            effective += int(decision["effective"])
        if effective > 1:
            raise ProtocolError("decision")
    elif kind == 2:
        fields(payload, {"kind", "points"})
        point_list(payload["points"])
        if not payload["points"]:
            raise ProtocolError("empty_metrics")
    elif kind == 3:
        fields(payload, {"kind", "severity", "body"})
        integer(payload["severity"], 1, 4)
        text(payload["body"], MAX_TEXT, False)
    elif kind == 4:
        fields(payload, {"kind", "name", "start_monotonic_ns", "end_monotonic_ns", "status", "reason"})
        text(payload["name"], MAX_ID, False)
        start = integer(payload["start_monotonic_ns"])
        end = integer(payload["end_monotonic_ns"])
        if (not correlation["span_id"] or time["monotonic_time_ns"] is None
                or not start <= end <= time["monotonic_time_ns"]):
            raise ProtocolError("span_time")
        integer(payload["status"], 1, 4)
        text(payload["reason"])
    elif kind == 5:
        fields(payload, {"kind", "resource_id", "measurements"})
        text(payload["resource_id"], MAX_ID, False)
        point_list(payload["measurements"])
        if not payload["measurements"]:
            raise ProtocolError("empty_measurements")
    elif kind == 6:
        fields(payload, {"kind", "role", "artifact"})
        text(payload["role"], MAX_ID, False)
        artifact = fields(payload["artifact"], {"algorithm", "digest", "size_bytes",
                                               "logical_name", "media_type", "producer"})
        if artifact["algorithm"] != "sha256":
            raise ProtocolError("artifact_algorithm")
        digest(artifact["digest"])
        integer(artifact["size_bytes"], 0, 1 << 40)
        text(artifact["logical_name"], MAX_TEXT, False)
        for key in ("media_type", "producer"):
            text(artifact[key], MAX_ID, False)
    return value


def symbolic_value(point: dict[str, Any]) -> float:
    """Convert neutral nonfinite status only at an official binary-proto edge."""
    return {1: lambda: float(point["value"]), 2: lambda: float("nan"),
            3: lambda: float("inf"), 4: lambda: -float("inf")}[point["status"]]()


def native_id(domain: str, value: str, size: int) -> bytes:
    """Map opaque native IDs to backend widths and retain originals in attrs."""
    result = hashlib.sha256((domain + "\0" + value).encode("utf-8")).digest()[:size]
    return result if any(result) else b"\0" * (size - 1) + b"\1"


class TensorBoardBackend:
    """Official text preserves lifecycles; scalar tensors preserve binary64."""

    def __init__(self, output: Path):
        from tensorboard.summary.writer.event_file_writer import EventFileWriter
        from tensorboard.compat.proto import event_pb2, summary_pb2, tensor_pb2, types_pb2
        from tensorboard.plugins.text.summary_v2 import text_pb
        self._event = event_pb2
        self._summary = summary_pb2
        self._tensor = tensor_pb2
        self._types = types_pb2
        self._text_summary = text_pb
        # A fresh exclusive output directory belongs to this one supervised child.
        output.mkdir(parents=False, exist_ok=False)
        self._writer = EventFileWriter(str(output), max_queue_size=128, flush_secs=10)

    def prepare(self, value: dict[str, Any]) -> Any | None:
        payload = value["payload"]
        wall = value["time"]["wall_time_unix_ns"]
        if payload["kind"] not in (1, 2, 5) or wall is None:
            return None
        lifecycle = payload["kind"] == 1
        points = (payload["metrics"] if lifecycle else
                  payload["points"] if payload["kind"] == 2 else payload["measurements"])
        if ((not lifecycle and (not points or any(point["status"] == 5 for point in points)))
                or any(point["step"]["value"] > MAX_I64 for point in points)
                or (lifecycle and value["identity"]["sequence"] > MAX_I64)):
            return None
        result = []
        if lifecycle:
            # Text is the complete lifecycle projection, including exact clocks,
            # identities, decisions, nonfinite statuses and absent observations.
            # It is not a reconstructed callback message or a metadata subset.
            series = {"stream_id": value["identity"]["stream_id"],
                      "project_id": value["correlation"]["project_id"],
                      "run_id": value["correlation"]["run_id"],
                      "attempt_id": value["correlation"]["attempt_id"],
                      "lifecycle_name": payload["name"]}
            suffix = hashlib.sha256(compact(series).encode("utf-8")).hexdigest()
            description = compact({"schema": "jyppx.tensorboard.lifecycle/v1",
                                   "series": series, "step_domain": "telemetry.sequence",
                                   "encoding": "complete-envelope-json",
                                   "missing_metric_projection": "text-only; no scalar value exists"})
            summary = self._text_summary(f"jyppx/lifecycle/{suffix}", compact(value),
                                         description=description)
            result.append(self._event.Event(wall_time=wall / 1_000_000_000,
                                            step=value["identity"]["sequence"], summary=summary))
        for point in points:
            # A lifecycle's complete text explicitly records status/reason for
            # missing observations. TensorBoard has no missing scalar value;
            # never turn that absence into zero or NaN. Standalone scalar/resource
            # batches lack the complete text projection and remain unsupported.
            if point["status"] == 5:
                continue
            definition = point["definition"]
            # The stable series identity contains the complete definition and
            # exact context; a step domain is never silently merged with another.
            series = {"definition": definition, "context": point["context"],
                      "step_kind": point["step"]["kind"],
                      "resource_id": payload.get("resource_id", "")}
            suffix = hashlib.sha256(compact(series).encode("utf-8")).hexdigest()
            tag = f"jyppx/scalar/{suffix}"
            description = compact({"series": series,
                                   "wall_time_unix_ns": wall,
                                   "original_metric_id": definition["metric_id"]})
            metadata = self._summary.SummaryMetadata(
                summary_description=description,
                plugin_data=self._summary.SummaryMetadata.PluginData(plugin_name="scalars"))
            tensor = self._tensor.TensorProto(dtype=self._types.DT_DOUBLE,
                                             double_val=[symbolic_value(point)])
            summary = self._summary.Summary(value=[self._summary.Summary.Value(
                tag=tag, tensor=tensor, metadata=metadata)])
            result.append(self._event.Event(wall_time=wall / 1_000_000_000,
                                            step=point["step"]["value"], summary=summary))
        return result

    def export(self, prepared: Any) -> None:
        for event in prepared:
            self._writer.add_event(event)
        # Observer completion, without an fsync/durability claim.
        self._writer.flush()

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ExportError("redirect_rejected")


class OtlpHttpBackend:
    """Official generated OTLP requests, bounded direct HTTP, explicit config."""

    def __init__(self, base_url: str, allowed_host: str, timeout_seconds: float,
                 resource_instance_id: str,
                 clock_mappings: dict[str, dict[str, int]] | None = None):
        from opentelemetry.proto.collector.metrics.v1 import metrics_service_pb2
        from opentelemetry.proto.collector.logs.v1 import logs_service_pb2
        from opentelemetry.proto.collector.trace.v1 import trace_service_pb2
        self._schemas = {"metrics": metrics_service_pb2, "logs": logs_service_pb2,
                         "traces": trace_service_pb2}
        parsed = urllib.parse.urlsplit(base_url)
        if (parsed.scheme not in ("http", "https") or not parsed.hostname
                or parsed.hostname != allowed_host or parsed.username or parsed.password
                or parsed.query or parsed.fragment or parsed.path not in ("", "/")
                or (parsed.scheme == "http" and parsed.hostname not in ("127.0.0.1", "::1"))):
            raise ProtocolError("collector_configuration")
        try:
            parsed.port
        except ValueError as error:
            raise ProtocolError("collector_port") from error
        if not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 30:
            raise ProtocolError("collector_timeout")
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        # The runner explicitly declares the source instance. Event/trace IDs
        # are never automatically promoted into metric resource dimensions.
        self._resource_instance_id = text(resource_instance_id, MAX_ID, False)
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
        self._mappings = clock_mappings or {}
        if len(self._mappings) > 64:
            raise ProtocolError("clock_mapping_capacity")
        for clock, anchor in self._mappings.items():
            text(clock, MAX_ID, False)
            fields(anchor, {"monotonic_ns", "unix_ns"})
            integer(anchor["monotonic_ns"])
            integer(anchor["unix_ns"])

    @staticmethod
    def _attributes(target: Any, values: dict[str, Any]) -> None:
        for key, value in sorted(values.items()):
            attribute = target.add()
            attribute.key = key
            if type(value) is bool:
                attribute.value.bool_value = value
            elif type(value) is int and -(1 << 63) <= value <= MAX_I64:
                attribute.value.int_value = value
            else:
                attribute.value.string_value = value if isinstance(value, str) else compact(value)

    @staticmethod
    def _base_attributes(value: dict[str, Any]) -> dict[str, Any]:
        return {"jyppx.identity": compact(value["identity"]),
                "jyppx.correlation": compact(value["correlation"]),
                "jyppx.attributes": compact(value["attributes"]),
                "jyppx.reliability": value["reliability"],
                "jyppx.source_schema": 1,
                "jyppx.wall_time_unix_ns": str(value["time"]["wall_time_unix_ns"])}

    def _resource(self, resource: Any, value: dict[str, Any], correlation: bool = True) -> None:
        attributes = {"service.name": "jyppx-training-framework",
                      "service.instance.id": self._resource_instance_id}
        if correlation:
            attributes.update({"jyppx.project_id": value["correlation"]["project_id"],
                               "jyppx.run_id": value["correlation"]["run_id"],
                               "jyppx.attempt_id": value["correlation"]["attempt_id"]})
        self._attributes(resource.attributes, attributes)

    def prepare(self, value: dict[str, Any]) -> Any | None:
        payload = value["payload"]
        if payload["kind"] in (2, 5) or (payload["kind"] == 1 and payload["metrics"]):
            measured = self._metrics(value)
            audit = self._log(value)
            if measured is None or audit is None:
                return None
            # Native event/trace/run identities, sample counts, status/reason,
            # and the exact producer step are audit-log data. Promoting these
            # changing values into metric dimensions would create one series
            # per observation and silently explode cardinality.
            return [measured, audit]
        prepared = self._span(value) if payload["kind"] == 4 else self._log(value)
        return None if prepared is None else [prepared]

    def _metrics(self, value: dict[str, Any]) -> Any | None:
        wall = value["time"]["wall_time_unix_ns"]
        if wall is None or wall < 0:
            return None
        module = self._schemas["metrics"]
        request = module.ExportMetricsServiceRequest()
        resource_metrics = request.resource_metrics.add()
        self._resource(resource_metrics.resource, value, correlation=False)
        scope = resource_metrics.scope_metrics.add()
        scope.scope.name, scope.scope.version = "jyppx.telemetry", "1"
        payload = value["payload"]
        points = (payload["metrics"] if payload["kind"] == 1 else
                  payload["points"] if payload["kind"] == 2 else payload["measurements"])
        if not points:
            return None
        units = {1: "1", 2: "{count}", 3: "s", 4: "By", 5: "1", 6: "%",
                 7: "{count}/s", 8: "By/s", 9: ""}
        for point in points:
            definition = point["definition"]
            output = scope.metrics.add()
            output.name = "jyppx.scalar." + definition["definition_digest"]
            output.description = definition["metric_id"]
            output.unit = units[definition["unit"]]
            sample = output.gauge.data_points.add()
            sample.time_unix_nano = wall
            attributes = {"jyppx.metric_definition": compact(definition),
                          "jyppx.metric_context": compact(point["context"]),
                          "jyppx.step_kind": point["step"]["kind"],
                          "jyppx.resource_id": payload.get("resource_id", "")}
            self._attributes(sample.attributes, attributes)
            if point["status"] == 5:
                # Official DataPointFlags.NO_RECORDED_VALUE_MASK; no invented 0.
                sample.flags = 1
            else:
                sample.as_double = symbolic_value(point)
        return "metrics", request

    def _log(self, value: dict[str, Any]) -> Any | None:
        module = self._schemas["logs"]
        request = module.ExportLogsServiceRequest()
        resource_logs = request.resource_logs.add()
        self._resource(resource_logs.resource, value)
        scope = resource_logs.scope_logs.add()
        scope.scope.name, scope.scope.version = "jyppx.telemetry", "1"
        record = scope.log_records.add()
        wall = value["time"]["wall_time_unix_ns"]
        if wall is not None and wall < 0:
            return None
        if wall is not None:
            record.time_unix_nano = wall
        payload = value["payload"]
        severity = payload["severity"] if payload["kind"] == 3 else 2
        record.severity_number = {1: 5, 2: 9, 3: 13, 4: 17}[severity]
        record.severity_text = {1: "DEBUG", 2: "INFO", 3: "WARN", 4: "ERROR"}[severity]
        record.body.string_value = compact(payload)
        attributes = self._base_attributes(value)
        attributes["jyppx.timestamp_missing"] = wall is None
        # Keep the complete neutral envelope alongside the human-facing payload
        # body, including monotonic clock provenance and exact absence semantics.
        attributes["jyppx.envelope"] = compact(value)
        self._attributes(record.attributes, attributes)
        correlation = value["correlation"]
        if correlation["trace_id"]:
            record.trace_id = native_id("jyppx.trace/v1", correlation["trace_id"], 16)
        if correlation["span_id"]:
            record.span_id = native_id("jyppx.span/v1", correlation["span_id"], 8)
        return "logs", request

    def _span(self, value: dict[str, Any]) -> Any | None:
        anchor = self._mappings.get(value["time"]["clock_id"])
        if anchor is None:
            return None
        payload = value["payload"]
        start = anchor["unix_ns"] + payload["start_monotonic_ns"] - anchor["monotonic_ns"]
        end = anchor["unix_ns"] + payload["end_monotonic_ns"] - anchor["monotonic_ns"]
        if not 0 <= start <= end <= MAX_U64:
            return None
        module = self._schemas["traces"]
        request = module.ExportTraceServiceRequest()
        resource_spans = request.resource_spans.add()
        self._resource(resource_spans.resource, value)
        scope = resource_spans.scope_spans.add()
        scope.scope.name, scope.scope.version = "jyppx.telemetry", "1"
        span = scope.spans.add()
        correlation = value["correlation"]
        span.trace_id = native_id("jyppx.trace/v1", correlation["trace_id"], 16)
        span.span_id = native_id("jyppx.span/v1", correlation["span_id"], 8)
        if correlation["parent_span_id"]:
            span.parent_span_id = native_id("jyppx.span/v1", correlation["parent_span_id"], 8)
        span.name = payload["name"]
        span.kind = 1  # Official Span.SPAN_KIND_INTERNAL.
        span.start_time_unix_nano, span.end_time_unix_nano = start, end
        span.status.code = {1: 0, 2: 1, 3: 2, 4: 2}[payload["status"]]
        span.status.message = payload["reason"]
        attributes = self._base_attributes(value)
        attributes.update({"jyppx.clock_id": value["time"]["clock_id"],
                           "jyppx.clock_mapping": compact(anchor),
                           "jyppx.span_status": payload["status"]})
        self._attributes(span.attributes, attributes)
        return "traces", request

    def export(self, prepared: Any) -> None:
        # A metrics+audit pair may be acknowledged only partially. Retrying the
        # immutable source is intentionally at least once at collector edges.
        for item in prepared:
            self._send(item)

    def _send(self, prepared: Any) -> None:
        signal, message = prepared
        payload = message.SerializeToString(deterministic=True)
        if len(payload) > MAX_LINE:
            raise ExportError("otlp_request_capacity")
        request = urllib.request.Request(self._base_url + "/v1/" + signal, data=payload,
                                         headers={"Content-Type": "application/x-protobuf"}, method="POST")
        try:
            with self._opener.open(request, timeout=self._timeout) as response:
                if response.status != 200:
                    raise ExportError("collector_status")
                data = response.read(1024 * 1024 + 1)
                if len(data) > 1024 * 1024:
                    raise ExportError("collector_reply_capacity")
        except (OSError, urllib.error.URLError) as error:
            raise ExportError("collector_io") from error
        module = self._schemas[signal]
        response_class = getattr(module, {"metrics": "ExportMetricsServiceResponse",
                                          "logs": "ExportLogsServiceResponse",
                                          "traces": "ExportTraceServiceResponse"}[signal])
        try:
            reply = response_class.FromString(data)
        except Exception as error:
            raise ExportError("collector_reply_proto") from error
        partial = reply.partial_success
        rejected = getattr(partial, {"metrics": "rejected_data_points",
                                      "logs": "rejected_log_records",
                                      "traces": "rejected_spans"}[signal])
        if rejected or partial.error_message:
            raise ExportError("collector_partial_success")

    def flush(self) -> None:
        # export waits for collector acknowledgement; no local queue is hidden.
        return None

    def close(self) -> None:
        return None


class Engine:
    """Serial protocol state; all batch identities preflight before effects."""

    def __init__(self, name: str, backend: Any, max_seen: int = MAX_SEEN,
                 max_seen_bytes: int = MAX_SEEN_BYTES):
        self.name, self.backend = name, backend
        self.max_seen, self.max_seen_bytes = max_seen, max_seen_bytes
        self.seen: dict[tuple[str, str], tuple[str, str, int]] = {}
        self.seen_bytes = 0
        self.counters = {"exported_events": 0, "sampled_out_events": 0, "unsupported_events": 0}
        self.closed = False

    def handle(self, request: Any) -> dict[str, Any]:
        if not isinstance(request, dict):
            raise ProtocolError("request")
        operation = request.get("operation")
        fields(request, {"protocol_version", "backend", "request_id", "operation"}
                        | ({"events"} if operation == "emit" else set()))
        integer(request["protocol_version"], 1, 1)
        if request["backend"] != self.name or operation not in ("emit", "flush", "close"):
            raise ProtocolError("backend_or_operation")
        request_id = text(request["request_id"], 64, False)
        response = {"protocol_version": 1, "request_id": request_id}
        if operation in ("flush", "close"):
            if request_id != operation:
                raise ProtocolError("request_identity")
            if not self.closed:
                self.backend.flush() if operation == "flush" else self.backend.close()
                self.closed = operation == "close"
        else:
            if self.closed:
                raise ProtocolError("closed")
            events = request["events"]
            if not isinstance(events, list) or not 1 <= len(events) <= MAX_EVENTS:
                raise ProtocolError("event_capacity")
            items = []
            staged: dict[tuple[str, str], tuple[str, str]] = {}
            staged_bytes = 0
            ordered_digests = []
            for item in events:
                fields(item, {"content_digest", "envelope"})
                content_digest = digest(item["content_digest"])
                value = envelope(item["envelope"])
                identity = value["identity"]
                key = (identity["stream_id"], identity["idempotency_key"])
                fingerprint = hashlib.sha256(compact(value).encode("utf-8")).hexdigest()
                old = self.seen.get(key) or staged.get(key)
                if old is not None and old[:2] != (content_digest, fingerprint):
                    raise ProtocolError("operation_conflict")
                if old is None:
                    staged[key] = content_digest, fingerprint
                    staged_bytes += sum(len(part.encode("utf-8")) for part in key) + 128
                ordered_digests.append(content_digest)
                items.append((key, content_digest, fingerprint, value))
            expected = hashlib.sha256(("jyppx.telemetry.sidecar-batch/v1"
                                       + "".join(ordered_digests)).encode("utf-8")).hexdigest()
            if request_id != expected:
                raise ProtocolError("request_identity")
            if (len(self.seen) + len(staged) > self.max_seen
                    or self.seen_bytes + staged_bytes > self.max_seen_bytes):
                raise ProtocolError("replay_capacity")
            # Materialize every projection before the first external write.
            prepared = [self.backend.prepare(item[3]) if item[0] not in self.seen else None
                        for item in items]
            receipts = []
            for (key, content_digest, fingerprint, _), output in zip(items, prepared):
                if key in self.seen:
                    disposition = self.seen[key][2]
                    receipts.append(DUPLICATE if disposition == ACCEPTED else disposition)
                    continue
                if output is None:
                    disposition = UNSUPPORTED
                    self.counters["unsupported_events"] += 1
                else:
                    self.backend.export(output)
                    disposition = ACCEPTED
                    self.counters["exported_events"] += 1
                self.seen[key] = content_digest, fingerprint, disposition
                self.seen_bytes += sum(len(part.encode("utf-8")) for part in key) + 128
                receipts.append(disposition)
            response["receipts"] = receipts
        response["counters"] = self.counters.copy()
        return response


def serve(engine: Engine, source: BinaryIO, destination: BinaryIO) -> int:
    """Process bounded newline frames; diagnostics never contain child input."""
    while True:
        raw = source.readline(MAX_LINE + 2)
        if not raw:
            return 0
        request_id = "invalid"
        fatal_frame = len(raw) > MAX_LINE + 1 or not raw.endswith(b"\n")
        try:
            if fatal_frame:
                raise ProtocolError("line_frame")
            request = strict_json(raw[:-1])
            if isinstance(request, dict):
                candidate = request.get("request_id")
                if isinstance(candidate, str) and len(candidate) <= 64 and candidate.isascii() and candidate.isalnum():
                    request_id = candidate
            response = engine.handle(request)
        except ProtocolError:
            response = {"protocol_version": 1, "request_id": request_id, "error": "invalid_request"}
        except Exception:
            # A successful prefix may already exist. The runner retains durable
            # originals and can retry; no raw exception, URL, or credential leaks.
            response = {"protocol_version": 1, "request_id": request_id, "error": "observer_failed"}
        output = compact(response).encode("utf-8")
        if len(output) > MAX_RESPONSE:
            return 3
        destination.write(output + b"\n")
        destination.flush()
        if fatal_frame:
            return 2


def check_dependencies(backend: str) -> None:
    if sys.version_info[:2] != (3, 11):
        raise ProtocolError("python_version")
    needed = ["protobuf", "tensorboard" if backend == "tensorboard" else "opentelemetry-proto"]
    for package in needed:
        if importlib.metadata.version(package) != OFFICIAL_VERSIONS[package]:
            raise ProtocolError("dependency_version")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("tensorboard", "opentelemetry"), required=True)
    parser.add_argument("--tensorboard-output", type=Path)
    parser.add_argument("--otlp-base-url")
    parser.add_argument("--allowed-host")
    parser.add_argument("--resource-instance-id")
    parser.add_argument("--http-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--clock-mappings", type=Path)
    args = parser.parse_args()
    try:
        check_dependencies(args.backend)
        if args.backend == "tensorboard":
            if (args.tensorboard_output is None or args.otlp_base_url or args.allowed_host
                    or args.clock_mappings or args.resource_instance_id):
                raise ProtocolError("backend_configuration")
            backend = TensorBoardBackend(args.tensorboard_output)
        else:
            if (not args.otlp_base_url or not args.allowed_host or not args.resource_instance_id
                    or args.tensorboard_output):
                raise ProtocolError("backend_configuration")
            mappings = {}
            if args.clock_mappings:
                with args.clock_mappings.open("rb") as stream:
                    mappings = strict_json(stream.read(64 * 1024 + 1), 64 * 1024)
            backend = OtlpHttpBackend(args.otlp_base_url, args.allowed_host,
                                      args.http_timeout_seconds, args.resource_instance_id, mappings)
        return serve(Engine(args.backend, backend), sys.stdin.buffer, sys.stdout.buffer)
    except Exception:
        # Startup diagnostics are stable and redacted, separate from stdout.
        sys.stderr.write("jyppx telemetry sidecar: startup failed\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
