# COLD Data

Convert the fixed raw COLD splits into the project quadruple format before building prompts:

```powershell
.\.venv\Scripts\python.exe data\cold_adapter.py `
  --raw-dir data\cold\raw `
  --output-dir data\cold\std
```

The adapter reads `train.csv`, `dev.csv` or `val.csv`, and `test.csv`.
It writes `data/cold/std/train.json`, `data/cold/std/val.json`, and `data/cold/std/test.json`.
Each sample is normalized to `id`, `content`, and one conservative quadruple.

Single-file conversion is still available with `--input`, but that mode may split the input when no split column is present.
