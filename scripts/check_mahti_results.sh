#!/bin/bash
# Check benchmark results on Mahti
# Usage: bash scripts/check_mahti_results.sh

echo "=== SLURM Job Status ==="
ssh mahti "squeue -u vtoivone -o '%.18i %.9P %.20j %.2t %.10M %.12l %.6D %R'" 2>&1

echo ""
echo "=== Recent BFCL Results ==="
ssh mahti "for f in \$(ls -t /scratch/project_2013932/vtoivone/pocket-agent-cli/data/results/bfcl/*/bfcl_results.json 2>/dev/null | head -10); do
    dir=\$(dirname \$f)
    timestamp=\$(basename \$dir)
    echo \"--- \$timestamp ---\"
    python3 -c \"
import json
d=json.load(open('\$f'))
for model_id, model_data in d.get('models',{}).items():
    print(f'  {model_id}: {model_data.get(\"full_match\",\"?\")}/{model_data.get(\"total\",\"?\")} ({model_data.get(\"full_match_pct\",\"?\")}%)')
    per_cat = model_data.get('per_category', {})
    for cat, cat_data in per_cat.items():
        print(f'    {cat}: {cat_data.get(\"full_match\",\"?\")}/{cat_data.get(\"total\",\"?\")} ({cat_data.get(\"full_match_pct\",\"?\")}%)')
\" 2>/dev/null
done"

echo ""
echo "=== Recent WebSearch QA Results ==="
ssh mahti "for f in \$(ls -t /scratch/project_2013932/vtoivone/pocket-agent-cli/data/results/websearch_qa/*/websearch_results.json 2>/dev/null | head -5); do
    dir=\$(dirname \$f)
    timestamp=\$(basename \$dir)
    echo \"--- \$timestamp ---\"
    python3 -c \"
import json
d=json.load(open('\$f'))
for r in d.get('results', []):
    model = r.get('model', '?')
    arch = r.get('architecture', '?')
    net = r.get('network_condition', '?')
    f1 = r.get('avg_f1', 0)
    em = r.get('avg_em', 0)
    print(f'  {model} | {arch:8s} | {net:15s} | F1={f1:.3f} EM={em:.3f}')
\" 2>/dev/null
done"

echo ""
echo "=== Job Output Files ==="
ssh mahti "ls -lt ~/bfcl-*-*.out ~/ws-*-*.out ~/all-bench*-*.out 2>/dev/null | head -10"
