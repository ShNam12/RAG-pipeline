"""Run selected rag1 stages in order and keep a complete per-run log."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Sequence, TextIO


ROOT = Path(__file__).resolve().parents[1]
STAGES = ("layout", "ocr", "tables", "chunk", "index")
OUTPUT_LABELS = {
    "layout": "Layout: ",
    "ocr": "OCR JSON: ",
    "tables": "Tables JSON: ",
    "chunk": "Chunks JSON: ",
}


class PipelineError(RuntimeError):
    """A stage or its required artifact failed."""


def parse_stages(value: str) -> tuple[str, ...]:
    """Accept a nonempty subset in pipeline order."""
    stages = tuple(part.strip() for part in value.split(","))
    if not stages or any(stage not in STAGES for stage in stages):
        raise ValueError(f"Stages must be a comma-separated subset of {', '.join(STAGES)}")
    if len(set(stages)) != len(stages) or stages != tuple(sorted(stages, key=STAGES.index)):
        raise ValueError("Stages must be unique and in pipeline order")
    return stages


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", type=parse_stages, default=STAGES,
                        help="comma-separated stages in order (default: all)")
    parser.add_argument("--source", type=Path, help="PDF input when layout runs")
    parser.add_argument("--layout-json", type=Path, help="existing layout.json when layout is skipped")
    parser.add_argument("--ocr-json", type=Path, help="existing ocr.json when OCR is skipped")
    parser.add_argument("--tables-json", type=Path,
                        help="existing tables.json to embed when tables is skipped")
    parser.add_argument("--chunks-json", type=Path, help="existing chunks.json when chunk is skipped")
    parser.add_argument("--env-file", type=Path, help="environment file passed to uv run")
    parser.add_argument("--log-file", type=Path, help="append full stage output to this log file")
    parser.add_argument("--allow-partial", action="store_true",
                        help="continue when layout, OCR, or table status is partial")
    parser.add_argument("--layout-output-dir", type=Path)
    parser.add_argument("--layout-image-dir", type=Path)
    parser.add_argument("--layout-dpi", type=int)
    parser.add_argument("--layout-device")
    parser.add_argument("--ocr-output-dir", type=Path)
    parser.add_argument("--ocr-image-dir", type=Path)
    parser.add_argument("--ocr-device")
    parser.add_argument("--ocr-manifest", type=Path)
    parser.add_argument("--table-output-dir", type=Path)
    parser.add_argument("--table-model")
    parser.add_argument("--table-device")
    parser.add_argument("--table-manifest", type=Path)
    parser.add_argument("--region-id", help="one table region ID")
    parser.add_argument("--collection", help="Qdrant collection for embedding and upload")
    return parser


def _absolute(path: Path) -> Path:
    return path.resolve()


def _required(path: Path | None, description: str) -> Path:
    if path is None:
        raise PipelineError(f"{description} is required")
    resolved = _absolute(path)
    if not resolved.is_file():
        raise PipelineError(f"{description} does not exist: {resolved}")
    return resolved


def _log(log: TextIO, message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    line = f"{timestamp} {message}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "backslashreplace").decode("ascii"), flush=True)
    print(line, file=log, flush=True)


def _status(path: Path, *, allow_partial: bool) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PipelineError(f"Cannot read artifact {path}: {error}") from error
    if not isinstance(payload, dict):
        raise PipelineError(f"Artifact root must be an object: {path}")
    status = payload.get("status", payload.get("ocr_status"))
    if status not in {"complete", "partial", "failed"}:
        raise PipelineError(f"Artifact has no recognized status: {path}")
    if status == "failed":
        raise PipelineError(f"Artifact status is failed: {path}")
    if status == "partial" and not allow_partial:
        raise PipelineError(f"Artifact status is {status}: {path}. Use --allow-partial to continue.")
    return payload


def _optional(command: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        command.extend((flag, str(value)))


def run_stage(stage: str, command: list[str], log: TextIO) -> Path | None:
    """Stream one CLI stage to the console and log, returning its output artifact."""
    _log(log, f"[{stage}] command: {json.dumps(command, ensure_ascii=False)}")
    started = perf_counter()
    marker = OUTPUT_LABELS.get(stage)
    output_path: Path | None = None
    with subprocess.Popen(
        command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    ) as process:
        assert process.stdout is not None
        for raw in process.stdout:
            line = raw.rstrip("\r\n")
            _log(log, f"[{stage}] {line}")
            if marker and line.startswith(marker):
                output_path = Path(line[len(marker):].strip())
        exit_code = process.wait()
    _log(log, f"[{stage}] exit={exit_code} elapsed_seconds={perf_counter() - started:.2f}")
    if exit_code != 0:
        raise PipelineError(f"Stage {stage} failed with exit code {exit_code}")
    if marker:
        if output_path is None:
            raise PipelineError(f"Stage {stage} did not report its output artifact")
        if not output_path.is_absolute():
            output_path = ROOT / output_path
        if not output_path.is_file():
            raise PipelineError(f"Stage {stage} reported a missing artifact: {output_path}")
        return output_path.resolve()
    return None


def run_pipeline(args: argparse.Namespace, log: TextIO) -> None:
    stages: tuple[str, ...] = args.stages
    if args.layout_json is not None and "layout" in stages:
        raise PipelineError("--layout-json is only for runs that skip layout")
    if args.ocr_json is not None and "ocr" in stages:
        raise PipelineError("--ocr-json is only for runs that skip OCR")
    if args.tables_json is not None and "tables" in stages:
        raise PipelineError("--tables-json is only for runs that skip tables")
    if args.chunks_json is not None and "chunk" in stages:
        raise PipelineError("--chunks-json is only for runs that skip chunk")

    source = _required(args.source, "--source") if "layout" in stages else None
    layout = (
        _required(args.layout_json, "--layout-json")
        if "layout" not in stages and any(stage in stages for stage in ("ocr", "tables"))
        else None
    )
    ocr = (
        _required(args.ocr_json, "--ocr-json")
        if "ocr" not in stages and any(stage in stages for stage in ("tables", "chunk"))
        else None
    )
    chunks = (
        _required(args.chunks_json, "--chunks-json")
        if "index" in stages and "chunk" not in stages else None
    )
    tables = _required(args.tables_json, "--tables-json") if args.tables_json else None
    env_file = _required(args.env_file, "--env-file") if args.env_file else None

    if layout is not None:
        _status(layout, allow_partial=args.allow_partial)
    if ocr is not None:
        _status(ocr, allow_partial=args.allow_partial)
    if tables is not None:
        _status(tables, allow_partial=args.allow_partial)
    if chunks is not None:
        _status(chunks, allow_partial=args.allow_partial)

    uv = ["uv", "run"]
    _optional(uv, "--env-file", env_file.as_posix() if env_file is not None else None)
    _log(log, f"Selected stages: {', '.join(stages)}")

    if "layout" in stages:
        assert source is not None
        command = [*uv, "rag1", "layout", str(source)]
        _optional(command, "--output-dir", args.layout_output_dir)
        _optional(command, "--image-dir", args.layout_image_dir)
        _optional(command, "--dpi", args.layout_dpi)
        _optional(command, "--device", args.layout_device)
        layout = run_stage("layout", command, log)
        assert layout is not None
        _status(layout, allow_partial=args.allow_partial)

    if "ocr" in stages:
        assert layout is not None
        command = [*uv, "rag1", "ocr", str(layout), "--model", "paddleocr-vl"]
        _optional(command, "--output-dir", args.ocr_output_dir)
        _optional(command, "--image-dir", args.ocr_image_dir)
        _optional(command, "--device", args.ocr_device)
        _optional(command, "--manifest", args.ocr_manifest)
        ocr = run_stage("ocr", command, log)
        assert ocr is not None
        _status(ocr, allow_partial=args.allow_partial)

    if "tables" in stages:
        assert layout is not None and ocr is not None
        regions = _status(layout, allow_partial=args.allow_partial).get("regions", [])
        has_tables = any(
            isinstance(region, dict)
            and str(region.get("raw_label", "")).strip().lower() == "table"
            and isinstance(region.get("location"), dict)
            and region["location"].get("type") == "visual"
            for region in regions
        )
        if not has_tables and args.region_id is None:
            _log(log, "[tables] No visual data-table proposals; skipping table reconstruction")
        else:
            command = [*uv, "--extra", "tabular", "rag1", "tables", str(layout),
                       "--ocr-json", str(ocr)]
            _optional(command, "--output-dir", args.table_output_dir)
            _optional(command, "--model", args.table_model)
            _optional(command, "--device", args.table_device)
            _optional(command, "--manifest", args.table_manifest)
            _optional(command, "--region-id", args.region_id)
            tables = run_stage("tables", command, log)
            assert tables is not None
            _status(tables, allow_partial=args.allow_partial)

    if "chunk" in stages:
        assert ocr is not None
        chunks = run_stage("chunk", [*uv, "rag1", "chunk", str(ocr)], log)
        assert chunks is not None

    if "index" in stages:
        assert chunks is not None
        command = [*uv, "rag1", "index", str(chunks)]
        _optional(command, "--tables-json", tables)
        _optional(command, "--collection", args.collection)
        run_stage("index", command, log)

    _log(log, "Pipeline completed")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log_path = args.log_file or ROOT / "logs" / f"pipeline-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}.log"
    log_path = _absolute(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        _log(log, f"Log file: {log_path}")
        try:
            run_pipeline(args, log)
        except (OSError, PipelineError, ValueError) as error:
            _log(log, f"Pipeline failed: {error}")
            return 1
        except KeyboardInterrupt:
            _log(log, "Pipeline interrupted")
            return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
