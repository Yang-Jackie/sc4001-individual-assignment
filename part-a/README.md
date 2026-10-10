# HDB price baseline

## Run it

Install the exact environment recorded in `uv.lock`:

```bash
uv sync
```

Run the baseline with:

```bash
uv run python src/baseline.py
```

Run these commands from the `part-a` directory. The input CSV and output directory are constants at the top of `src/baseline.py`, while the data split, features, model size, and training hyperparameters are defined in `src/config.py`.

## Notebooks

Open the notebooks in VS Code or Jupyter and select the Python interpreter in
`part-a/.venv`. Run all cells from top to bottom, with the working directory set
to either `part-a` or the repository root. Keep `hdb_price_prediction.csv` in
`part-a`.

- `a1.ipynb`: Ray grid search and final baseline retraining, using seed 42.
- `a2.ipynb`: baseline, continuous-only, and Sigmoid experiments, using seeds
  42, 43, 44, 45, and 46. It runs independently of A1, with A1's selected hidden
  width 256, embedding dimension 4, and six epochs. Adam uses learning rate
  0.01 and batch size 512 for all runs.
- `a3.ipynb`: wide-and-deep lambda search over 0, 0.25, 0.5, 1, and 2, using
  seed 42 and the selected A1 architecture. It selects by best 2021 validation
  RMSE with A1's early-stopping rules, then retrains on 2017–2021 for the selected
  epoch count. It evaluates the selected model and the six-epoch A1 baseline
  using seeds 42, 43, 44, 45, and 46. This notebook also runs independently.

A2 fits preprocessing on 2017–2021 and evaluates each model once on 2022 after
training. It writes per-seed results and the summary to `outputs/a2/`, and saves
`a2_test_rmse.png` and `a2_test_r2.png` beside the notebook. The summary and plots
use sample standard deviation (`ddof=1`). The notebooks contain the final run
outputs.

A3 saves its lambda search, validation history, selection, per-seed results,
and summary to `outputs/a3/`. Its figures are `a3_validation_rmse.png` and
`a3_test_results.png`. Seed-42 baseline and wide-and-deep weights are saved in
`outputs/a3/` for A4; preprocessing can be recreated with `prepare_data` using
the same 2017–2021 training years. Both branches share the categorical
embeddings, and the combined prediction is converted to SGD only once.

## Code map

```text
src/config.py       run configuration
src/data.py         split, fit preprocessing, tensor conversion
src/model.py        categorical embeddings, MLP, and wide-and-deep model
src/baseline.py     single-run training, RMSE evaluation, and model export
```
