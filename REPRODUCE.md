# Reproducing every number

Every figure in the documentation comes from one of the commands below. They are ordered by cost.

Each block says what it produces and which document it fills. Run them in order. Nothing later depends on anything you skip, except that training needs a dataset and the organic comparison needs a trained model.

```bash
# ---------------------------------------------------------------
# SETUP
# ---------------------------------------------------------------
cd ~/Documents/Projects/clean && rm -rf reef-water-carbonate
cd ~/Documents/Projects/Reef_Water_Cleanup && bash make_clean_copies.sh

cd ~/Documents/Projects/clean/reef-water-carbonate
py -3 -m venv .venv
source .venv/Scripts/activate
pip install -e ".[dev,ml]"
python -c "import numpy, tensorflow, cleaned_reef_water; print('ok')"
```

To regenerate from nothing rather than trusting the committed results, wipe them first. Skip this if you only want to re-verify what is already here.

```bash
rm -rf data results
mkdir -p data results
```

## Fast: the gate

Under two minutes. Silence means everything passed.

```bash
# ---------------------------------------------------------------
# Nothing to record unless one of these fails.
# ---------------------------------------------------------------
python -m ruff check .
python -m mypy src/
python -m pytest -m "not tensorflow" -W "ignore::UserWarning"
python -m pytest -m tensorflow    -W "ignore::UserWarning"
```

```bash
# ---------------------------------------------------------------
# Recomputes every documented value checkable in seconds and
# reports where each is stated. Exits non-zero on disagreement,
# so it works as a release gate. Record the full output.
# ---------------------------------------------------------------
python scripts/verify_claims.py
```

```bash
# ---------------------------------------------------------------
# Worst-case agreement against an independent implementation
# across a grid of salinity and temperature. The -s flag prints
# the table. Feeds doc 17.
# ---------------------------------------------------------------
python -m pytest tests/test_pyco2sys_agreement.py -s
```

## Fast: identifiability and the accuracy floor

Seconds each.

```bash
# ---------------------------------------------------------------
# The pKa sweep and the twelve-state robustness table.
# Both feed doc 15.
# ---------------------------------------------------------------
python scripts/organic_identifiability.py --sweep
python scripts/organic_identifiability.py --sweep-states \
    --output results/organic_identifiability.json
```

```bash
# ---------------------------------------------------------------
# Cramer-Rao bound, the leverage table and the species grid.
# Fills the accuracy-floor tables in docs 10 and 16.
# ---------------------------------------------------------------
python scripts/information_limits.py --verify --seed 42 \
    --output results/limits.json
```

## Medium: the classical benchmark

Roughly ten minutes. Chunked, so a crash costs one chunk rather than the run.

```bash
# ---------------------------------------------------------------
# Six cases: two instrument grades against three nuisance
# configurations, on organic-free water. Replaces the benchmark
# table in doc 12 and the classical rows in doc 16.
# ---------------------------------------------------------------
bash scripts/resume.sh scripts/run_baseline.py \
    --seed 42 --chunk-size 20 --output results/baseline.json
```

```bash
# ---------------------------------------------------------------
# Supporting figures. Noise budget feeds doc 19; the feature
# baseline feeds the summary-statistic argument in doc 13.
# ---------------------------------------------------------------
python scripts/noise_budget.py --instrument laboratory
python scripts/noise_budget.py --instrument hobbyist
python scripts/feature_baseline.py --instrument hobbyist
```

## Long: data and training

Hours each. Both stages are resumable and safe to interrupt.

```bash
# ---------------------------------------------------------------
# Sixteen thousand curves, plus an organic-free control on its own
# seed. The control seed MUST differ from the dataset seed:
# sampling is deterministic, so reusing it would reproduce the same
# compositions and the control would be water the model trained on.
# ---------------------------------------------------------------
bash scripts/resume.sh scripts/generate_dataset.py \
    --instrument hobbyist --seed 42 --n-samples 16000 --chunk-size 500 \
    --output data/dataset_hobbyist.npz

bash scripts/resume.sh scripts/generate_dataset.py \
    --instrument hobbyist --organic-probability 0.0 --seed 43 \
    --n-samples 4000 --chunk-size 250 \
    --output data/dataset_hobbyist_control.npz
```

```bash
# ---------------------------------------------------------------
# Three runs differing only in initialisation. The split seed is
# held fixed so the spread measures training variance alone rather
# than confounding it with a different held-out set.
#
# Doc 16 reports the median across these three with the range.
# ---------------------------------------------------------------
for s in 42 43 44; do
  python scripts/train_autoencoder.py \
      --dataset data/dataset_hobbyist.npz \
      --split-seed 42 --train-seed $s \
      --output-dir results/seed$s
done
```

Write each run to its own `--output-dir`. Omitting it puts the result at `results/autoencoder_hobbyist.json`, where the next run overwrites it and the three-run median cannot be computed.

```bash
# ---------------------------------------------------------------
# Learned against classical on the same curves, across three
# populations. This is the only matched comparison in the project:
# identical inputs, identical point count, both methods in one pass.
# Fills the organic table in doc 16.
# ---------------------------------------------------------------
python scripts/compare_organic.py \
    --model results/seed42/autoencoder_best.keras \
    --normalizer results/seed42/autoencoder_normalizer.npz \
    --dataset data/dataset_hobbyist.npz \
    --control data/dataset_hobbyist_control.npz \
    --split-seed 42 --n-eval 600 \
    --output results/organic_comparison.json
```

## Reading the training runs back

The training output scrolls past. This prints everything doc 16 needs from the three saved files.

```bash
python - <<'PY'
import json, glob, os

for d in sorted(glob.glob("results/seed*")):
    path = os.path.join(d, "autoencoder_hobbyist.json")
    if not os.path.exists(path):
        print(f"{d}: no result yet")
        continue
    j = json.load(open(path))

    print(f"\n=== {d}   {j['epochs_run']} epochs")

    inf = j["inference"]
    print(f"  network {inf['network_ms_per_curve']:.2f} ms   "
          f"classical {inf['classical_ms_per_curve']:.1f} ms   "
          f"speedup {inf['speedup']:.0f}x")

    joint = j["joint_error_structure"]
    print(f"  corr(dA,dC) {joint['corr_dA_dC']:+.3f}   "
          f"sd(dA-dC) {joint['sd_dA_minus_dC']:.2f}")

    print("  species, umol/kg:")
    for name, m in j["metrics_umol_per_kg"].items():
        print(f"    {name:11s} bias {m['bias']:+8.2f}  sd {m['sd']:7.2f}  "
              f"MAE {m['mae']:7.2f}  median {m['median_abs']:7.2f}  "
              f"max {m['max_abs']:8.2f}")

    print("  latent recovery:")
    for name, v in j["nuisance_recovery"].items():
        print(f"    {name:14s} bias {v['bias']:+.5f}  resid sd {v['sd']:.5f}  "
              f"true sd {v['true_sd']:.5f}  recovery {v['recovery_ratio']:.2f}")
PY
```

The residual spread is stored as `sd`, and the timings live under `inference` rather than at the top level. Reaching for `residual_sd` or scanning the top level for `ms_per_curve` returns nothing and looks like a failed run.

The median and range across the three runs, which is what doc 16 actually reports:

```bash
python - <<'PY'
import json, glob, os, statistics as st

runs = [json.load(open(os.path.join(d, "autoencoder_hobbyist.json")))
        for d in sorted(glob.glob("results/seed*"))
        if os.path.exists(os.path.join(d, "autoencoder_hobbyist.json"))]
print(f"{len(runs)} runs\n")

def report(label, get):
    vals = sorted(get(r) for r in runs)
    print(f"  {label:22s} median {st.median(vals):8.2f}   "
          f"range {vals[0]:.2f} to {vals[-1]:.2f}")

for q in ("alkalinity", "dic", "hco3", "co3", "boh4"):
    for stat in ("bias", "sd", "mae", "median_abs", "max_abs"):
        report(f"{q} {stat}", lambda r, q=q, s=stat: r["metrics_umol_per_kg"][q][s])
    print()

for name in ("slope", "offset", "total_organic", "pk_organic", "pump_scale"):
    report(f"{name} recovery",
           lambda r, n=name: r["nuisance_recovery"][n]["recovery_ratio"])
print()
report("corr(dA,dC)", lambda r: r["joint_error_structure"]["corr_dA_dC"])
report("sd(dA-dC)",   lambda r: r["joint_error_structure"]["sd_dA_minus_dC"])
report("classical ms", lambda r: r["inference"]["classical_ms_per_curve"])
report("network ms",   lambda r: r["inference"]["network_ms_per_curve"])
PY
```

## Figures

```bash
pip install -e ".[notebook]"
jupyter lab notebooks/curves.ipynb
```

The chemistry sections need only the package. The results sections read `results/*.json` and skip themselves cleanly if those files are absent, so the notebook runs from a bare checkout with no data.

To re-execute every cell without opening it:

```bash
python -m jupyter nbconvert --to notebook --execute --inplace \
    notebooks/curves.ipynb
```

## Final check

```bash
# ---------------------------------------------------------------
# Record only a failure. Anything else means the documentation and
# the code agree.
# ---------------------------------------------------------------
python -m ruff check .
python scripts/verify_claims.py
```

## What is not reproducible here

The kinetic feasibility result quoted in doc 20 was measured in a separate project and is not recomputable from this repository. It is stated as a conclusion.

Everything else in the documentation is produced by a command above.
