#!/usr/bin/env python3
"""Self-daemonizing 10k TMF921 dataset generator (mock backend)."""
import os
import sys
import platform

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(REPO_ROOT, 'output', '10k_generated')
LOG = os.path.join(OUTPUT_DIR, 'generate.log')
PIDFILE = os.path.join(OUTPUT_DIR, 'pid.txt')

os.makedirs(OUTPUT_DIR, exist_ok=True)

if platform.system() == 'Windows':
    print("Error: daemon mode is not supported on Windows.")
    print("Use generate_10k.py directly instead.")
    sys.exit(1)

# Fork to background
pid = os.fork()
if pid > 0:
    print(f"Daemon started, child PID: {pid}")
    sys.exit(0)

os.setsid()
pid = os.fork()
if pid > 0:
    sys.exit(0)

# Redirect output to log
sys.stdout.flush()
sys.stderr.flush()
log = open(LOG, 'w', buffering=1)
os.dup2(log.fileno(), 1)
os.dup2(log.fileno(), 2)

os.chdir(REPO_ROOT)

with open(PIDFILE, 'w') as f:
    f.write(str(os.getpid()))

# Force mock backend BEFORE importing tmf921_dataset_gen
os.environ['INFERENCE_BACKEND'] = 'mock'

sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))
from pathlib import Path
from tmf921_dataset_gen.config import Settings
from tmf921_dataset_gen.graph.workflow import run_generation
from datetime import datetime, timezone
import json
import time


def append_jsonl(target_path: Path, source_path: Path) -> None:
    if not source_path.exists():
        return
    payload = source_path.read_text(encoding='utf-8')
    if not payload:
        return
    if target_path.exists() and target_path.stat().st_size > 0:
        with target_path.open('rb') as handle:
            handle.seek(-1, 2)
            if handle.read(1) != b'\n':
                with target_path.open('a', encoding='utf-8') as writer:
                    writer.write('\n')
    with target_path.open('a', encoding='utf-8') as writer:
        writer.write(payload)
        if not payload.endswith('\n'):
            writer.write('\n')


def build_combined_manifest(output_base: Path, total_generated: int, generation_method: str) -> dict:
    batch_manifests = []
    quality_scores = []
    semantic_scores = []
    tio_scores = []
    diversity_scores = []

    for manifest_path in sorted(output_base.glob('batch_*/manifest.json')):
        payload = json.loads(manifest_path.read_text(encoding='utf-8'))
        batch_manifests.append({'path': str(manifest_path), 'record_count': payload.get('record_count', 0)})
        metrics = payload.get('quality_metrics', {})
        if 'average_quality_score' in metrics:
            quality_scores.append(metrics['average_quality_score'])
        if 'average_semantic_score' in metrics:
            semantic_scores.append(metrics['average_semantic_score'])
        if 'average_tio_compliance' in metrics:
            tio_scores.append(metrics['average_tio_compliance'])
        if 'diversity_score' in metrics:
            diversity_scores.append(metrics['diversity_score'])

    def average(values: list[float]) -> float | None:
        if not values:
            return None
        return round(sum(values) / len(values), 4)

    return {
        'generation_summary': {
            'total_samples': total_generated,
            'generation_method': generation_method,
            'generation_timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
            'batch_count': len(batch_manifests),
        },
        'quality_metrics': {
            'average_quality_score': average(quality_scores),
            'average_semantic_score': average(semantic_scores),
            'average_tio_compliance': average(tio_scores),
            'average_diversity_score': average(diversity_scores),
        },
        'batches': batch_manifests,
    }


def main():
    output_base = Path('output/10k_generated')
    output_base.mkdir(parents=True, exist_ok=True)
    combined_file = output_base / 'dataset.jsonl'
    batch_size = 1000

    settings = Settings.from_env()
    print(f"Backend: {settings.inference_backend}", flush=True)
    print(f"Start: {datetime.now(timezone.utc).isoformat()}", flush=True)

    combined_file.write_text("")
    total_generated = 0

    for batch_num in range(1, 11):
        batch_dir = output_base / f'batch_{batch_num}'
        batch_dir.mkdir(exist_ok=True)
        batch_file = batch_dir / 'dataset.jsonl'

        if batch_file.exists():
            existing = sum(1 for line in batch_file.read_text().splitlines() if line.strip())
            if existing >= batch_size:
                print(f"Batch {batch_num}: skip ({existing} done)", flush=True)
                append_jsonl(combined_file, batch_file)
                total_generated += existing
                continue

        print(f"=== Batch {batch_num}/10 ===", flush=True)
        print(f"Time: {datetime.now(timezone.utc).isoformat()}", flush=True)
        start = time.time()
        try:
            records = run_generation(settings, count=batch_size, output_dir=batch_dir)
            elapsed = time.time() - start
            total_generated += len(records)
            print(f"Batch {batch_num}: {len(records)} samples in {elapsed:.1f}s (total: {total_generated})", flush=True)
            if batch_file.exists():
                append_jsonl(combined_file, batch_file)
        except Exception as e:
            print(f"ERROR batch {batch_num}: {e}", flush=True)
            import traceback
            traceback.print_exc()

    manifest = build_combined_manifest(output_base, total_generated, 'heuristic_templates_mock_backend')
    (output_base / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f"Done! Total: {total_generated}", flush=True)

main()
