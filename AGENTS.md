# Repository Guidelines

## Project Structure & Module Organization

This repository contains Vietnamese hate-speech detection experiments and a demo app.

- `app/`: Streamlit demo and shared runtime code. `app.py` owns the UI; `model_runtime.py` loads artifacts and runs inference.
- `app/requirements.txt`: Python dependencies for the demo app.
- `dataset/`: dataset notes, Hugging Face download examples, and label mappings.
- `notebooks/models/baselines/`: baseline experiment notebooks and saved outputs.
- `notebooks/models/proposed/`: proposed ViAmpleHate/PhoBERT notebooks and outputs.
- `docs/`: design notes, plans, and comparison images.
- `paper/`: paper references and related work material.

Keep large trained artifacts under the matching notebook `output/` directory. Avoid committing downloaded raw datasets or cache directories.

## Build, Test, and Development Commands

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r app\requirements.txt
streamlit run app\app.py
```

These commands create a local environment, install app dependencies, and launch the demo. The first run may download Hugging Face models such as `vinai/phobert-base`, so internet access may be required.

For notebook work, start Jupyter from the repository root so relative output paths remain stable:

```powershell
jupyter notebook notebooks
```

## Coding Style & Naming Conventions

Use Python 3 with 4-space indentation, type hints where they clarify interfaces, and `pathlib.Path` for repository-relative paths. Follow the existing app pattern: constants in `UPPER_SNAKE_CASE`, functions in `snake_case`, and model identifiers as short lowercase strings. Keep Vietnamese user-facing text encoded as UTF-8 and verify it renders correctly in Streamlit and notebooks.

Notebook names should describe dataset, model family, and purpose, for example `vihsd-baseline-tf-idf-lr-svm.ipynb`.

## Testing Guidelines

No automated test suite is currently defined. Before submitting app changes, run:

```powershell
streamlit run app\app.py
```

Manually verify model selection, CPU/CUDA mode handling, empty-input warnings, and at least one prediction path. For notebook changes, rerun the changed notebook cells needed to regenerate affected metrics or output images, and document any intentional metric changes.

## Commit & Pull Request Guidelines

Recent commit messages use short imperative prefixes such as `add:` and `update`, for example `add: academic skill for paper writting` and `update streamlit app`. Keep commits focused and use concise messages like `update app inference cache` or `add voz proposed notebook`.

Pull requests should include a brief purpose, changed datasets/models, commands or notebooks run, and screenshots for Streamlit UI changes. Link related issues or design notes in `docs/` when applicable, and call out large artifacts or model downloads explicitly.

## Security & Configuration Tips

Do not commit secrets, local `.venv/`, Hugging Face caches, or raw private datasets. Prefer documented dataset loaders in `dataset/README.md` and keep generated files in predictable `output/` folders.

## Codex Skill Notes

Claude academic skills have been migrated into `.codex/skills/` for Codex-oriented use. Prefer `AGENTS.md` for repository guidance, and load only the relevant migrated skill when working on research, paper writing, peer review, or academic pipeline tasks. The original `.claude/` directory is preserved for compatibility.
