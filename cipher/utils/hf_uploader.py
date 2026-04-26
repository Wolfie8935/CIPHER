"""
cipher/utils/hf_uploader.py

Hugging Face Hub integration — upload trained model zips and manage
CIPHER artifacts on HuggingFace Hub.

Usage:
    python -m cipher.utils.hf_uploader --repo wolfie8935/cipher-specialists
"""
from __future__ import annotations

import argparse
import os
import zipfile
from datetime import datetime
from pathlib import Path


HF_REPO_ID = os.getenv("HF_REPO_ID", "wolfie8935/cipher-specialists")
HF_TRACES_REPO = os.getenv("HF_TRACES_REPO", "wolfie8935/cipher-traces")
# Public Space URL (for dashboards / health JSON; not used for Hub API calls)
HF_SPACE_URL = os.getenv(
    "HF_SPACE_URL",
    "https://huggingface.co/spaces/wolfie8935/cipher-openenv",
)


def _env_push_traces_enabled() -> bool:
    v = (os.getenv("CIPHER_PUSH_TRACES_HF") or "").strip().lower()
    return v in ("1", "true", "yes", "on")

SPECIALIST_DIRS = {
    "red_planner": os.getenv("RED_PLANNER_LORA_PATH", "red trained/cipher-red-planner-v1"),
    "red_analyst": os.getenv("RED_ANALYST_LORA_PATH", "red trained/cipher-red-analyst-v1"),
    "blue_surveillance": os.getenv("BLUE_SURVEILLANCE_LORA_PATH", "blue trained/cipher-blue-surveillance-v1"),
    "blue_threat_hunter": os.getenv("BLUE_THREAT_HUNTER_LORA_PATH", "blue trained/cipher-blue-threat-hunter-v1"),
}


def _get_api():
    """Return HuggingFace HfApi instance, or raise with a helpful message."""
    try:
        from huggingface_hub import HfApi
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        return HfApi(token=token)
    except ImportError:
        raise RuntimeError(
            "huggingface_hub is not installed. Run: pip install huggingface_hub"
        )


def _zip_model_dir(model_dir: str, zip_path: Path) -> bool:
    """Zip a model directory. Returns True on success."""
    src = Path(model_dir)
    if not src.exists():
        print(f"  [SKIP] {model_dir} — directory not found")
        return False
    print(f"  [ZIP]  {src} → {zip_path.name}")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in src.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(src.parent))
    return True


def upload_specialists(repo_id: str = HF_REPO_ID, dry_run: bool = False) -> dict:
    """
    Zip and upload all specialist LoRA models to Hugging Face Hub.

    Returns a dict mapping specialist name → upload status.
    """
    api = _get_api()
    tmp_dir = Path("tmp_hf_upload")
    tmp_dir.mkdir(exist_ok=True)
    results = {}

    # Ensure repo exists
    if not dry_run:
        try:
            from huggingface_hub import create_repo
            create_repo(repo_id, repo_type="model", exist_ok=True)
            print(f"  [HUB]  Repository: https://huggingface.co/{repo_id}")
        except Exception as e:
            print(f"  [WARN] Could not create repo: {e}")

    for name, model_dir in SPECIALIST_DIRS.items():
        zip_path = tmp_dir / f"{name}.zip"
        zipped = _zip_model_dir(model_dir, zip_path)
        if not zipped:
            results[name] = "skipped"
            continue

        if dry_run:
            print(f"  [DRY]  Would upload {zip_path.name} to {repo_id}")
            results[name] = "dry_run"
        else:
            try:
                api.upload_file(
                    path_or_fileobj=str(zip_path),
                    path_in_repo=f"specialists/{zip_path.name}",
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=f"Upload {name} specialist — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                )
                print(f"  [OK]   Uploaded {name}")
                results[name] = "uploaded"
            except Exception as e:
                print(f"  [ERR]  {name}: {e}")
                results[name] = f"error: {e}"

    # Clean up zips
    for f in tmp_dir.glob("*.zip"):
        f.unlink(missing_ok=True)

    return results


def upload_traces(traces_dir: str = "episode_traces",
                  repo_id: str = HF_TRACES_REPO,
                  dry_run: bool = False) -> int:
    """
    Upload episode traces JSON files to a Hugging Face Dataset repo.
    Returns count of uploaded files.
    """
    api = _get_api()
    src = Path(traces_dir)
    if not src.exists():
        print(f"  [SKIP] {traces_dir} directory not found")
        return 0

    trace_files = sorted(src.glob("*.json"))
    if not trace_files:
        print("  [SKIP] No trace files found")
        return 0

    if not dry_run:
        try:
            from huggingface_hub import create_repo
            create_repo(repo_id, repo_type="dataset", exist_ok=True)
            print(f"  [HUB]  Dataset: https://huggingface.co/datasets/{repo_id}")
        except Exception as e:
            print(f"  [WARN] Could not create dataset repo: {e}")

    uploaded = 0
    for tf in trace_files:
        if dry_run:
            print(f"  [DRY]  Would upload trace {tf.name}")
            uploaded += 1
        else:
            try:
                api.upload_file(
                    path_or_fileobj=str(tf),
                    path_in_repo=f"traces/{tf.name}",
                    repo_id=repo_id,
                    repo_type="dataset",
                    commit_message=f"Add trace {tf.name}",
                )
                uploaded += 1
                print(f"  [OK]   Uploaded {tf.name}")
            except Exception as e:
                print(f"  [ERR]  {tf.name}: {e}")

    return uploaded


def upload_episode_trace_file(
    trace_path: str | Path,
    *,
    repo_id: str | None = None,
    dry_run: bool = False,
) -> bool:
    """
    Upload a single episode trace JSON to the HF Dataset at ``traces/<filename>``.

    Used after each local save when ``CIPHER_PUSH_TRACES_HF`` is enabled.
    Requires ``HF_TOKEN`` (or ``HUGGINGFACE_TOKEN``) with write access to the dataset.
    """
    path = Path(trace_path)
    if not path.is_file() or path.suffix.lower() != ".json":
        return False
    rid = repo_id or HF_TRACES_REPO
    api = _get_api()

    if not dry_run:
        try:
            from huggingface_hub import create_repo

            create_repo(rid, repo_type="dataset", exist_ok=True)
        except Exception:
            pass

    dest = f"traces/{path.name}"
    if dry_run:
        print(f"  [DRY]  Would upload trace {path.name} → datasets/{rid}/{dest}")
        return True
    try:
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=dest,
            repo_id=rid,
            repo_type="dataset",
            commit_message=f"Add trace {path.name}",
        )
        print(f"  [OK]   Uploaded trace to Hub: datasets/{rid} (file {dest})")
        return True
    except Exception as e:
        print(f"  [ERR]  Trace upload {path.name}: {e}")
        return False


def maybe_push_episode_trace(trace_path: str | Path, *, dry_run: bool = False) -> bool:
    """
    If ``CIPHER_PUSH_TRACES_HF`` is truthy, upload ``trace_path`` to ``HF_TRACES_REPO``.
    Otherwise no-op (returns False).
    """
    if not _env_push_traces_enabled():
        return False
    return upload_episode_trace_file(trace_path, dry_run=dry_run)


def upload_reports(reports_dir: str = "episode_reports",
                   repo_id: str = HF_TRACES_REPO,
                   dry_run: bool = False) -> int:
    """Upload episode narrative reports to HF Dataset."""
    api = _get_api()
    src = Path(reports_dir)
    if not src.exists():
        print(f"  [SKIP] {reports_dir} not found")
        return 0

    report_files = sorted(src.glob("*.md"))
    if not report_files:
        print("  [SKIP] No report files found")
        return 0

    uploaded = 0
    for rf in report_files:
        if dry_run:
            print(f"  [DRY]  Would upload report {rf.name}")
            uploaded += 1
        else:
            try:
                api.upload_file(
                    path_or_fileobj=str(rf),
                    path_in_repo=f"reports/{rf.name}",
                    repo_id=repo_id,
                    repo_type="dataset",
                    commit_message=f"Add report {rf.name}",
                )
                uploaded += 1
                print(f"  [OK]   Uploaded {rf.name}")
            except Exception as e:
                print(f"  [ERR]  {rf.name}: {e}")

    return uploaded


def main():
    parser = argparse.ArgumentParser(description="CIPHER Hugging Face Hub uploader")
    parser.add_argument("--repo", default=HF_REPO_ID, help="HF model repo ID")
    parser.add_argument("--traces-repo", default=HF_TRACES_REPO, help="HF dataset repo for traces")
    parser.add_argument("--dry-run", action="store_true", help="List what would be uploaded without doing it")
    parser.add_argument("--skip-models", action="store_true", help="Skip model uploads")
    parser.add_argument("--skip-traces", action="store_true", help="Skip trace uploads")
    parser.add_argument("--skip-reports", action="store_true", help="Skip report uploads")
    args = parser.parse_args()

    print(f"\n{'='*58}")
    print("  CIPHER → Hugging Face Hub Uploader")
    print(f"  {'[DRY RUN] ' if args.dry_run else ''}Target: {args.repo}")
    print(f"{'='*58}\n")

    if not args.skip_models:
        print("── Specialist Models ──────────────────────────────────")
        upload_specialists(repo_id=args.repo, dry_run=args.dry_run)

    if not args.skip_traces:
        print("\n── Episode Traces ─────────────────────────────────────")
        n = upload_traces(repo_id=args.traces_repo, dry_run=args.dry_run)
        print(f"  Total traces: {n}")

    if not args.skip_reports:
        print("\n── Narrative Reports ──────────────────────────────────")
        n = upload_reports(repo_id=args.traces_repo, dry_run=args.dry_run)
        print(f"  Total reports: {n}")

    print("\n  Done.\n")


if __name__ == "__main__":
    main()
