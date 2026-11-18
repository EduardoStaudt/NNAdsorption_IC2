# NNAdsorption_IC2
Develoment my second Scitific Iniciation, aplly PIINs in structure of five adsorbates and two adsorbents.

Hybrid physics-informed neural network (TensorFlow) for fixed-bed adsorption with variable
velocity (solver v23b model, `docs/modelo_solver_v23b_preview.pdf`). Written for `n_comp`
gases; the current phases use 2 gases and 1 adsorbent.

- `docs/nondim.md`: dimensionless equations and groups
- `docs/architecture.md`: modules, network, loss, hooks for 5 gases / 2 adsorbents
- `docs/phase0_results.md`: Phase 0 (one case) results
- `docs/phase1_results.md`: Phase 1 (50 + 10 cases, provisional MOL data) results

## Environment (Windows: TensorFlow GPU needs WSL2)

```bash
python3.11 -m venv ~/ic2env
~/ic2env/bin/python -m pip install -r requirements.txt
```

## Usage (inside WSL2, from the repository root)

```bash
~/ic2env/bin/python -m pytest -q                                   # tests
~/ic2env/bin/python scripts/train_single_case.py                   # Phase 0: one case, forward problem
~/ic2env/bin/python scripts/make_mol_dataset.py --n-cases 60       # provisional Phase 1 data (MOL, not v23b)
~/ic2env/bin/python scripts/train_parametric.py --data <file.h5>   # Phase 1: several cases, hybrid loss
~/ic2env/bin/python scripts/evaluate.py --run runs/<run> [--data <holdout.h5>]
```

Datasets made by `make_mol_dataset.py` are tagged `provisório_MOL` (file name without the accent,
attribute `dataset_tag`): they come from the small MOL solver of this repository and must not be
mixed with the v23b dataset. `train_parametric.py` warns when it reads one and records the tag in
`summary.json`.

Runs are written to `runs/<name>/` (config, CSV log, checkpoints, weights, `summary.json`,
figures). Every number lives in `configs/*.yaml`.
