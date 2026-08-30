"""Versioned, deterministic metadata for cross-run training checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1
DEFAULT_ROOT = Path(__file__).resolve().parent / "checkpoints"
MANIFEST_NAME = "continuity-manifest.json"
MODEL_SUFFIXES = {".json", ".pt"}


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def checkpoint_files(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.name != MANIFEST_NAME and path.suffix in MODEL_SUFFIXES
    )


def build_manifest(root: Path, *, seed: int, run_id: str, source_sha: str) -> dict:
    files = checkpoint_files(root)
    if not files:
        raise ValueError(f"no checkpoint files found under {root}")
    versions = {}
    for package in ("torch", "numpy", "hisss"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "run_id": str(run_id),
        "source_sha": source_sha,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": versions,
        "files": [
            {"path": path.relative_to(root).as_posix(), "sha256": _digest(path), "bytes": path.stat().st_size}
            for path in files
        ],
    }


def write_manifest(root: Path, *, seed: int, run_id: str, source_sha: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    destination = root / MANIFEST_NAME
    destination.write_text(
        json.dumps(build_manifest(root, seed=seed, run_id=run_id, source_sha=source_sha), indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def validate_manifest(root: Path) -> dict:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValueError(f"missing {MANIFEST_NAME} in restored checkpoint artifact")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported checkpoint manifest schema: {payload.get('schema_version')!r}")
    if not isinstance(payload.get("seed"), int):
        raise ValueError("checkpoint manifest seed must be an integer")
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("checkpoint manifest contains no files")
    for item in files:
        relative = Path(str(item.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe checkpoint path: {relative}")
        candidate = root / relative
        if not candidate.is_file():
            raise ValueError(f"checkpoint file missing: {relative}")
        if _digest(candidate) != item.get("sha256"):
            raise ValueError(f"checkpoint digest mismatch: {relative}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("write", "validate"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--source-sha", default="")
    args = parser.parse_args()
    if args.command == "write":
        if args.seed is None:
            parser.error("--seed is required for write")
        path = write_manifest(args.root, seed=args.seed, run_id=args.run_id, source_sha=args.source_sha)
        print(path)
    else:
        payload = validate_manifest(args.root)
        print(json.dumps({"schema_version": payload["schema_version"], "seed": payload["seed"], "run_id": payload["run_id"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
