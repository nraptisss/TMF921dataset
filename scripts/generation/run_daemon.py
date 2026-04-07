#!/usr/bin/env python3
"""Self-daemonizing 10k TMF921 dataset generator (mock backend)."""
import os
import sys
import platform
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / 'output' / '10k_generated'
LOG = OUTPUT_DIR / 'generate.log'
PIDFILE = OUTPUT_DIR / 'pid.txt'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if platform.system() == 'Windows':
    print("Error: daemon mode is not supported on Windows.")
    print("Use scripts/generation/generate_10k.py directly instead.")
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

os.chdir(str(REPO_ROOT))

with open(PIDFILE, 'w') as f:
    f.write(str(os.getpid()))

# Force mock backend BEFORE importing tmf921_dataset_gen
os.environ['INFERENCE_BACKEND'] = 'mock'

sys.path.insert(0, str(REPO_ROOT / 'src'))
from tmf921_dataset_gen.config import Settings
from tmf921_dataset_gen.graph.workflow import run_generation
from common import append_jsonl, build_combined_manifest
from datetime import datetime, timezone
import json
import time


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


if __name__ == "__main__":
    main()
