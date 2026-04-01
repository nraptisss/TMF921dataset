#!/bin/bash

# TMF921 Dataset Generator - Fixed Scoring Version
# This script generates 10k samples with the fixed scoring (no artificial floors)
# Run this from your terminal in the codex-dataset directory

set -e

# Activate virtual environment
source .venv/bin/activate

# Create output directory
OUTPUT_DIR="output/10k_fixed_scoring"
mkdir -p $OUTPUT_DIR

echo "=============================================="
echo "TMF921 Dataset Generation"
echo "=============================================="
echo "Output directory: $OUTPUT_DIR"
echo "Scoring: Fixed (no artificial floors)"
echo "Target samples: 10,000"
echo ""

# Generate in batches to avoid timeout
BATCH_SIZE=1000
TOTAL_SAMPLES=10000
COMBINED_FILE="$OUTPUT_DIR/dataset.jsonl"

# Initialize combined file
> $COMBINED_FILE

for ((i=0; i<$TOTAL_SAMPLES; i+=$BATCH_SIZE)); do
    BATCH=$((i/$BATCH_SIZE + 1))
    BATCH_DIR="$OUTPUT_DIR/batch_$BATCH"
    mkdir -p $BATCH_DIR
    
    echo "=== Batch $BATCH/10 ==="
    echo "Generating $BATCH_SIZE samples..."
    
    # Generate batch
    python -m tmf921_dataset_gen.cli sample --count $BATCH_SIZE --out $BATCH_DIR
    
    # Append to combined file
    if [ -f "$BATCH_DIR/dataset.jsonl" ]; then
        cat "$BATCH_DIR/dataset.jsonl" >> $COMBINED_FILE
        BATCH_COUNT=$(wc -l < "$BATCH_DIR/dataset.jsonl")
        echo "  ✓ Generated $BATCH_COUNT samples"
    else
        echo "  ✗ Warning: Batch $BATCH failed"
    fi
    
    # Small delay between batches
    sleep 2
done

# Count total samples
TOTAL_GENERATED=$(wc -l < $COMBINED_FILE)
echo ""
echo "=============================================="
echo "Generation complete!"
echo "=============================================="
echo "Total samples: $TOTAL_GENERATED"
echo "Output file: $COMBINED_FILE"
echo ""

# Create manifest with fixed scoring notes
python -c "
import json
from datetime import datetime

manifest = {
    'generation_summary': {
        'total_samples': $TOTAL_GENERATED,
        'generation_method': 'enhanced_diversity_agent_with_fixed_scoring',
        'template_variations': ['conversational', 'administrative', 'business_focused', 'technical_precise'],
        'noise_injection': {
            'kpi_omission_rate': 0.1,
            'value_noise_rate': 0.05,
            'value_noise_magnitude': 0.05
        },
        'generation_timestamp': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'model_used': {
            'reasoning': 'Qwen/Qwen3.5-9B',
            'bulk': 'Qwen/Qwen3.5-4B',
            'embedding': 'BAAI/bge-large-en-v1.5'
        },
        'settings': {
            'inference_backend': 'local-transformers',
            'local_files_only': True,
            'local_device': 'cuda:0',
            'local_dtype': 'bfloat16'
        }
    },
    'scoring_fixes': {
        'semantic_score': 'Removed 0.75 floor in llm_judge',
        'realism_score': 'Removed 0.85 floor when anchored',
        'notes': 'Quality scores are now accurate without artificial boosting'
    }
}

with open('$OUTPUT_DIR/manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)

print('Manifest created: $OUTPUT_DIR/manifest.json')
"

echo ""
echo "=============================================="
echo "Next steps:"
echo "=============================================="
echo "1. Run the validation script:"
echo "   python scripts/validate_local.py $OUTPUT_DIR/dataset.jsonl"
echo ""
echo "2. Update the Hugging Face dataset"
echo "=============================================="

exit 0