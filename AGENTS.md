# Repository Guidelines

## Project Structure & Module Organization

This repository implements AerialVLN training, evaluation, and AirSim simulator integration. Main navigation code lives in `src/vlnce_src/`, with global CLI parameters in `src/common/param.py`. Model policies, encoders, losses, and the imitation-learning trainer are under `Model/`. AirSim RPC tools and simulator settings are in `airsim_plugin/`. Shared environment, graph, logging, and distributed utilities are in `utils/`. The embedded multimodal model stack is in `llava/`. Shell entry points are in `scripts/`, and static README assets are in `files/`. Ad hoc validation utilities currently live at the repository root, for example `scan_lmdb_actions.py`.

## Build, Test, and Development Commands

Create the environment and install dependencies:

```bash
conda create -n AirVLN python=3.8
conda activate AirVLN
pip install -r requirements.txt
pip install airsim==1.7.0
```

Run common workflows:

```bash
bash scripts/train.sh          # train baseline / current configured policy
bash scripts/eval.sh           # evaluate checkpoints
python scan_lmdb_actions.py    # scan LMDB labels for invalid actions
```

AirSim workflows require the external `DATA/` and `ENVs/` directories described in `README.md`, plus the simulator server started by the relevant script.

## Coding Style & Naming Conventions

Use Python 3 style with 4-space indentation. Prefer existing repository patterns over broad refactors. Keep CLI options in `src/common/param.py`, environment behavior in `src/vlnce_src/env.py` or `utils/env_utils.py`, and model-specific code in `Model/`. Use descriptive snake_case for functions, variables, and script names. Class names should use PascalCase. Avoid hard-coded absolute paths in new code; expose paths as arguments when practical.

## Testing Guidelines

There is no central test runner configured. For utility scripts, add a direct command-line smoke test and document expected output. For data issues, prefer small validation scripts that print counts, bad keys, and value ranges. Before training changes, run a short LMDB scan and a small training/evaluation subset when available, for example with reduced `--batchSize` or `--EVAL_NUM`.

## Commit & Pull Request Guidelines

Recent commits use short imperative summaries, for example `Add LMDB action scanner` or `Add llava policy`. Keep commits focused and avoid mixing generated data, checkpoints, or unrelated local edits. Pull requests should describe the changed training/evaluation path, list commands run, mention required datasets or simulator assumptions, and include logs or metrics when behavior changes.

## Security & Configuration Tips

Do not commit credentials, private dataset links, model checkpoints, generated LMDBs, or local machine-specific paths. Large artifacts belong outside the repository, typically under the workspace-level `DATA/` and `ENVs/` directories.
