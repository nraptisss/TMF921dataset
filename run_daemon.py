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
                with open(combined_file, 'a') as f:
                    f.write(batch_file.read_text())
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
                with open(combined_file, 'a') as f:
                    f.write(batch_file.read_text())
        except Exception as e:
            print(f"ERROR batch {batch_num}: {e}", flush=True)
            import traceback
            traceback.print_exc()

    manifest = {
        "generation_summary": {
            "total_samples": total_generated,
            "generation_method": "heuristic_templates_mock_backend",
            "generation_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    }
    (output_base / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f"Done! Total: {total_generated}", flush=True)

main()
