# Reproduce

## Environment
Python 3.12; `pip install -r requirements_v2.txt`; Ollama (versions in MANIFEST.md).
Pull models: `llama3.1:8b-instruct-q4_K_M`, `mistral:7b-instruct-v0.3-q4_K_M`,
`qwen2.5:7b-instruct-q4_K_M` (+ `qwen2.5:32b-instruct-q4_K_M` on Apple silicon).

## Verify the corpus
```
sha256sum corpus/abstracts_v2.jsonl        # Linux; compare to corpus/abstracts_v2.sha256
shasum -a 256 corpus/abstracts_v2.jsonl    # macOS
```

## Option A — regenerate tables/figures from the published logs (minutes)
```
unzip logs_v2.zip                # the archived logs (from the Zenodo deposit)
python analysis/analyze.py       # writes results/: summary.md, per_cell.csv, per_run.csv, figures
```

## Option B — rerun the experiments from scratch (hours–days)
```
# start Ollama with a fixed default context
#   Windows PowerShell:  $env:OLLAMA_CONTEXT_LENGTH="8192"; ollama serve
#   macOS:               OLLAMA_CONTEXT_LENGTH=8192 ollama serve

python -m pytest -q tests                                   # 9/9
python scripts/e0_validate.py                               # E0 RESULT: PASS
python run_matrix.py --exp E2  --config config/v2/machine_win.yaml
python run_matrix.py --exp E3  --config config/v2/machine_win.yaml
python run_matrix.py --exp E3b --config config/v2/machine_win.yaml
python run_matrix.py --exp E6  --config config/v2/machine_win.yaml
python run_matrix.py --exp E6o --config config/v2/machine_win.yaml
python run_matrix.py --exp E4  --config config/v2/machine_mac.yaml    # Apple silicon
python run_matrix.py --exp E5  --config config/v2/machine_mac.yaml
# set d2_min_run_words from the clean ROC (see the paper, Table 3), then:
python run_matrix.py --exp E6d2 --config config/v2/machine_win.yaml
python analysis/analyze.py
```

## Map results to the paper
- Table 1 ← `results/per_cell.csv` (E3, E4 rows)
- Table 2 ← `results/per_cell.csv` (E3b rows)
- Table 3 ← `results/per_cell.csv` (E6 / E6d2 / E6o rows)
- Figures 1–3 ← `results/fig_r0_heatmap.png`, `fig_it_ribbons.png`, `fig_dose_response.png`
- GEE model ← end of `results/summary.md`
