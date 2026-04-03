#!/bin/bash
# Generate 10k TMF921 dataset samples with local GPU model
# Run with: nohup bash scripts/generation/run_generate_10k.sh > output/generate_10k.log 2>&1 &

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate

OUTPUT_DIR="output/10k_gpu_generated"
BATCH_SIZE=1000
TOTAL_SAMPLES=10000
COMBINED_FILE="$OUTPUT_DIR/dataset.jsonl"

mkdir -p "$OUTPUT_DIR"

append_jsonl() {
    local source_file="$1"
    if [ ! -f "$source_file" ]; then
        return
    fi
    if [ -s "$COMBINED_FILE" ] && [ "$(tail -c 1 "$COMBINED_FILE" | wc -l)" -eq 0 ]; then
        printf '\n' >> "$COMBINED_FILE"
    fi
    cat "$source_file" >> "$COMBINED_FILE"
    if [ -s "$source_file" ] && [ "$(tail -c 1 "$source_file" | wc -l)" -eq 0 ]; then
        printf '\n' >> "$COMBINED_FILE"
    fi
}

echo "=============================================="
echo "TMF921 Dataset Generation - 10k samples"
echo "Start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Backend: local-transformers (GPU)"
echo "Model: Qwen/Qwen3.5-9B"
echo "=============================================="

# Initialize combined file
> "$COMBINED_FILE"

for ((i=0; i<$TOTAL_SAMPLES; i+=$BATCH_SIZE)); do
    BATCH_NUM=$((i/$BATCH_SIZE + 1))
    BATCH_DIR="$OUTPUT_DIR/batch_$BATCH_NUM"
    mkdir -p "$BATCH_DIR"

    echo ""
    echo "=== Batch $BATCH_NUM/10 ==="
    echo "Time: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "Generating $BATCH_SIZE samples..."

    # Skip if batch already completed
    if [ -f "$BATCH_DIR/dataset.jsonl" ]; then
        EXISTING=$(wc -l < "$BATCH_DIR/dataset.jsonl" 2>/dev/null || echo "0")
        if [ "$EXISTING" -ge "$BATCH_SIZE" ]; then
            echo "  Batch $BATCH_NUM already complete ($EXISTING samples), skipping..."
            append_jsonl "$BATCH_DIR/dataset.jsonl"
            continue
        fi
    fi

    # Generate batch with extended timeout
    python -m tmf921_dataset_gen.cli sample --count $BATCH_SIZE --out "$BATCH_DIR"

    # Append to combined file
    if [ -f "$BATCH_DIR/dataset.jsonl" ]; then
        append_jsonl "$BATCH_DIR/dataset.jsonl"
        BATCH_COUNT=$(wc -l < "$BATCH_DIR/dataset.jsonl")
        TOTAL_SO_FAR=$(wc -l < "$COMBINED_FILE")
        echo "  Batch done: $BATCH_COUNT samples (total so far: $TOTAL_SO_FAR)"
    else
        echo "  WARNING: Batch $BATCH_NUM produced no output file"
    fi
done

# Count total
TOTAL_GENERATED=$(wc -l < "$COMBINED_FILE")

echo ""
echo "=============================================="
echo "Generation complete!"
echo "End: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Total samples: $TOTAL_GENERATED"
echo "Output: $COMBINED_FILE"
echo "=============================================="

# Create manifest
python -c "
import json
from datetime import datetime
from pathlib import Path

output_dir = Path('$OUTPUT_DIR')
batch_manifests = []
for manifest_path in sorted(output_dir.glob('batch_*/manifest.json')):
    payload = json.loads(manifest_path.read_text())
    batch_manifests.append({
        'path': str(manifest_path),
        'record_count': payload.get('record_count', 0),
        'quality_metrics': payload.get('quality_metrics', {})
    })

manifest = {
    'generation_summary': {
        'total_samples': $TOTAL_GENERATED,
        'generation_method': 'gpu_local_transformers_qwen35_9b',
        'template_variations': ['conversational', 'administrative', 'business_focused', 'technical_precise'],
        'generation_timestamp': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'model_used': {
            'reasoning': 'Qwen/Qwen3.5-9B',
            'bulk': 'Qwen/Qwen3.5-9B',
            'embedding': 'HashingVectorizer (fallback)'
        },
        'settings': {
            'inference_backend': 'local-transformers',
            'local_files_only': True,
            'local_device': 'cuda:0',
            'local_dtype': 'bfloat16',
            'thinking_disabled': True
        },
        'batch_count': len(batch_manifests)
    },
    'batches': batch_manifests
}

with open('$OUTPUT_DIR/manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)

print('Manifest written to $OUTPUT_DIR/manifest.json')
"

echo "Done!"
exit 0
