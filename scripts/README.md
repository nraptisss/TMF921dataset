# Scripts

Operational helper scripts are grouped here to keep the repository root focused on source code and core entry points.

## Generation Helpers

Location: `generation/`

- `generate_10k.py`: foreground mock-backend 10k generation in 10 batches.
- `run_daemon.py`: daemonized variant of 10k mock-backend generation.
- `generate_batch.py`: single-batch helper for explicit count/output runs.
- `run_generate_10k.sh`: shell workflow for GPU-based 10k runs.
- `generate_fixed.sh`: legacy fixed-scoring batch workflow.

Run all scripts from repository root or via explicit path, for example:

```bash
python scripts/generation/generate_batch.py 20 output/gpu_test
```
