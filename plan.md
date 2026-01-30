# Refactor Plan (Package + Notebooks)

## Goal
- Package core logic under `src/` for clean imports.
- Keep outputs under `results/` by default, but allow custom output paths.
- Minimize notebook-specific hacks (e.g., `sys.path` edits).

## Steps
1. **Create package layout**: add `src/idr/` with `__init__.py` and move/clone modules into it.
2. **Split responsibilities**:
   - `idr/utils.py`: device + model loading/patching + `SequenceEvolver`.
   - `idr/preparation.py`: gene dictionaries + fetch helpers + sorting.
   - `idr/sequence.py`: sequence generation helpers.
   - `idr/embedding.py`: embedding helpers.
   - `idr/io.py`: load/save helpers.
3. **Update results path handling**: add `results_dir` argument with `Path` resolution; default `results/` but user override allowed.
4. **Update notebooks**: replace `sys.path` hacks with package imports; adapt to new module names.
5. **Smoke check**: ensure imports resolve and paths are coherent.
