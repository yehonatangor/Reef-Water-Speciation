# Reproducible pipeline for cleaned_reef_water.
#
# Every target is deterministic: seeds are fixed in the scripts and recorded in
# the output JSON alongside the package version, Python version and platform.
#
#   make install     install the package with dev + ml extras
#   make check       lint, type-check, test and verify docs (the release gate)
#   make verify      recompute every documented number that runs in seconds
#   make baseline    benchmark the classical least-squares inverse
#   make limits      Cramer-Rao bounds: what accuracy is achievable at all
#   make data        generate synthetic datasets (no TensorFlow)
#   make train       train the direct CNN regressor (reference)
#   make autoencoder train the physics-constrained autoencoder (recommended)
#   make compare     learned vs classical on organic and organic-free water
#   make identifiability  how much organic alkalinity is measurable at all
#   make all         check + baseline + train
#   make clean       remove build and result artefacts

PYTHON ?= python3
RESULTS ?= results

.PHONY: install check verify lint types test baseline baseline-quick limits data train autoencoder compare identifiability train-laboratory all clean dist

install:
	$(PYTHON) -m pip install -e ".[dev,ml]"

lint:
	$(PYTHON) -m ruff check .

types:
	$(PYTHON) -m mypy src/

# Two processes, deliberately.  TensorFlow and SciPy's optimisers do not tear
# down reliably together: the suite passes and then the interpreter dumps core
# on the way out, giving a non-zero exit status with no failing test.  The
# `tensorflow` marker is applied automatically by conftest.py.
# Exit status 5 means "no tests collected", which is what happens when the ml
# extra is not installed and there is no TensorFlow to test.  That is not a
# failure; anything else is.
test:
	$(PYTHON) -m pytest -m "not tensorflow" -W "ignore::UserWarning"
	@$(PYTHON) -m pytest -m tensorflow -W "ignore::UserWarning"; \
	status=$$?; \
	if [ $$status -eq 5 ]; then \
		echo "  no TensorFlow installed - TF tests not run (pip install -e '.[ml]')"; \
	elif [ $$status -ne 0 ]; then \
		exit $$status; \
	fi

check: lint types test verify

# Recompute every documented number that can be checked in seconds, and report
# where each one is stated and where it came from.  Exits non-zero if the
# documentation and the code disagree.
verify:
	$(PYTHON) scripts/verify_claims.py

# Full benchmark: 6 cases (2 instrument grades x 3 nuisance models).
# Takes roughly 10 minutes on a laptop; use baseline-quick to smoke-test.
#
# Chunked and looped because scipy's least_squares segfaults the interpreter
# intermittently on Windows -- not input-dependent, so the only reliable
# mitigation is a fresh process and a record of what is already done.  Each
# curve is seeded independently of the chunking, so the merged result is
# identical to a single-process run; verified in tests/test_run_baseline.py.
# On a machine that does not crash the loop simply runs once.
#
# The loop retries ONLY on "chunks remain" (exit 2) or a signal death
# (>=128, i.e. the segfault).  Any other status is a real error -- a missing
# dependency, a bad argument -- and must stop, or a plain ImportError spins
# the shell forever printing the same traceback.
baseline:
	@i=0; \
	while [ $$i -lt 200 ]; do \
		$(PYTHON) scripts/run_baseline.py --n-curves 100 --n-points 100 \
			--chunk-size 20 --max-chunks-per-run 1 \
			--output $(RESULTS)/baseline.json; \
		status=$$?; \
		if [ $$status -eq 0 ]; then exit 0; fi; \
		if [ $$status -ne 2 ] && [ $$status -lt 128 ]; then \
			echo "run_baseline.py failed with status $$status - not a crash, stopping"; \
			exit $$status; \
		fi; \
		i=$$((i + 1)); \
	done; \
	echo "gave up after 200 attempts"; exit 1

# The information-theoretic floor.  --verify asserts the bound does not exceed
# the measured baseline; if it does, the bound is mis-specified.
limits:
	$(PYTHON) scripts/information_limits.py --verify \
		--output $(RESULTS)/limits.json

baseline-quick:
	$(PYTHON) scripts/run_baseline.py --n-curves 10 --n-points 40 \
		--output $(RESULTS)/baseline_quick.json

# Data generation and training run in SEPARATE processes.  generate_dataset.py
# never imports TensorFlow: TF and SciPy's optimisers do not coexist reliably in
# one interpreter on Windows.
#
# Chunking is retained for resumability rather than for crash tolerance: this is
# hours of compute, and an interruption should cost one chunk rather than the
# run.  A single-process run works too -- drop --max-chunks-per-run.
data:
	while ! $(PYTHON) scripts/generate_dataset.py --instrument hobbyist \
		--chunk-size 500 --max-chunks-per-run 1; do : ; done
	while ! $(PYTHON) scripts/generate_dataset.py --instrument laboratory \
		--chunk-size 500 --max-chunks-per-run 1; do : ; done

train: data/dataset_hobbyist.npz
	$(PYTHON) scripts/train_cnn.py --dataset data/dataset_hobbyist.npz \
		--output-dir $(RESULTS)

# The recommended architecture: CNN encoder -> physical latents ->
# differentiable chemistry.  The value is amortised inference -- one forward
# pass instead of an optimisation per sample -- and an organic alkalinity term
# the classical inverse does not have.  Accuracy on ordinary seawater is
# expected to MATCH the classical fit, not beat it; see docs/information_limits.md.
autoencoder: data/dataset_hobbyist.npz
	$(PYTHON) scripts/train_autoencoder.py --dataset data/dataset_hobbyist.npz \
		--output-dir $(RESULTS)

# The organic-free control set, needed by `make compare`.  Evaluation only, so
# it is smaller than the training sets.
data/dataset_hobbyist_control.npz:
	$(PYTHON) scripts/generate_dataset.py --instrument hobbyist \
		--organic-probability 0.0 --seed 20260817 --n-samples 4000 \
		--output $@

# Learned vs classical on organic and organic-free water.  Runs the classical
# fits in a subprocess; --split-seed MUST match the training run.
compare: data/dataset_hobbyist_control.npz
	$(PYTHON) scripts/compare_organic.py \
		--model $(RESULTS)/autoencoder_best.keras \
		--normalizer $(RESULTS)/autoencoder_normalizer.npz \
		--dataset data/dataset_hobbyist.npz \
		--control data/dataset_hobbyist_control.npz \
		--split-seed 42 --n-eval 600 \
		--output $(RESULTS)/organic_comparison.json

# How much of organic alkalinity a titration can resolve at all.  Pure
# chemistry: no dataset, no model, seconds to run.
identifiability:
	$(PYTHON) scripts/organic_identifiability.py --sweep
	$(PYTHON) scripts/organic_identifiability.py --sweep-states \
		--output $(RESULTS)/organic_identifiability.json

train-laboratory: data/dataset_laboratory.npz
	$(PYTHON) scripts/train_cnn.py --dataset data/dataset_laboratory.npz \
		--output-dir $(RESULTS)

data/dataset_%.npz:
	$(PYTHON) scripts/generate_dataset.py --instrument $*

all: check baseline data train

dist:
	$(PYTHON) -m build

clean:
	rm -rf build dist $(RESULTS) data .pytest_cache .mypy_cache .ruff_cache \
		src/*.egg-info htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
