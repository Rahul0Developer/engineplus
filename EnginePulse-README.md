# EnginePulse

**A clean, tested, reproducible toolkit for Remaining Useful Life (RUL) prediction on NASA C-MAPSS**

[![CI](https://github.com/yourusername/enginepulse/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/enginepulse/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

Most RUL tutorials on the C-MAPSS dataset leak data, mix train and test engines, and report inflated numbers.  
EnginePulse fixes that.

It gives you:

1. **One-command data loading** with correct, leakage-free preprocessing  
2. **Fair benchmark** of classical and deep models using the proper NASA metrics  
3. **Ready FastAPI service** so you can serve predictions immediately  
4. **Optional document Q&A** over maintenance PDFs (with citations)

You can use it to learn, to compare your own models, or as a solid starting point for real predictive-maintenance work.

---

## Why this exists

The famous NASA C-MAPSS turbofan dataset is the standard benchmark for remaining-useful-life prediction.  
Almost every public notebook has one or more of these problems:

- Scales using the full dataset (including test engines)
- Creates sliding windows that cross engine boundaries
- Uses random train/test splits instead of engine-ID splits
- Reports only RMSE and ignores the NASA scoring function

EnginePulse was built to be the clean reference implementation that students, researchers, and engineers can trust.

---

## Quick Start

```bash
# Install
pip install enginepulse

# Download and prepare data (FD001 by default)
enginepulse download
enginepulse preprocess

# Run the full benchmark
enginepulse benchmark

# Start the prediction API
enginepulse serve
```

That’s it. You now have correct preprocessing, a fair comparison table, and a running API.

---

## Installation

```bash
# Basic (classical models only)
pip install enginepulse

# With deep learning support
pip install "enginepulse[deep]"

# With document Q&A
pip install "enginepulse[docqa]"

# Everything
pip install "enginepulse[all]"
```

Or install from source:

```bash
git clone https://github.com/yourusername/enginepulse.git
cd enginepulse
pip install -e ".[all]"
```

---

## Supported Models

| Model              | Type       | Notes                          |
|--------------------|------------|--------------------------------|
| Ridge              | Classical  | Strong baseline                |
| XGBoost            | Classical  | Usually competitive             |
| 1D-CNN             | Deep       | Fast and effective             |
| GRU / LSTM         | Deep       | Sequence modelling             |
| Transformer (opt.) | Deep       | Behind a flag                  |

All models share the same interface:

```python
from enginepulse.models import get_model

model = get_model("xgb")
model.fit(X_train, y_train)
preds = model.predict(X_test)
model.save("artifacts/xgb.joblib")
```

---

## Metrics

Every experiment reports three numbers:

- **RMSE**
- **MAE**
- **NASA Score** (asymmetric scoring function that heavily penalises late predictions)

This is the combination used in the original C-MAPSS papers and most serious follow-up work.

---

## Adding Your Own Model

```python
from enginepulse.models.base import BaseRULModel

class MyModel(BaseRULModel):
    def fit(self, X, y):
        ...
    def predict(self, X):
        ...
    def save(self, path):
        ...
    def load(self, path):
        ...
```

Register it and it automatically appears in the benchmark and CLI.

---

## API

```bash
enginepulse serve
```

Endpoints:

- `GET  /health`
- `GET  /models`
- `POST /predict` — send a sensor window, get RUL (+ optional uncertainty)

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sensors": [[...], [...], ...], "model": "xgb"}'
```

---

## Document Q&A (Optional)

```bash
pip install "enginepulse[docqa]"
```

- Ingest your own maintenance PDFs (public-domain or openly licensed only)
- Heading-aware chunking
- Local vector store (Chroma)
- Retrieval-only mode (no API key required)
- Optional LLM mode for full answers with citations

---

## Project Structure

```
src/enginepulse/
├── data/          # download, load, preprocess, windows
├── models/        # base class + classical + deep
├── metrics/       # RMSE, MAE, NASA score
├── train/         # trainer + benchmark harness
├── api/           # FastAPI application
├── docqa/         # optional RAG module
└── cli.py
```

---

## Reproducibility

- All experiments are driven by YAML configs
- Random seeds are fixed
- Git commit hash and config are stamped into every result file
- Tests catch the common leakage and preprocessing mistakes

---

## Citation

If you use this toolkit in academic work, please cite the original C-MAPSS dataset:

> A. Saxena and K. Goebel (2008).  
> “Turbofan Engine Degradation Simulation Data Set”,  
> NASA Ames Prognostics Data Repository,  
> NASA Ames Research Center, Moffett Field, CA.

And feel free to mention EnginePulse.

---

## Disclaimer

This is a research and educational toolkit.  
It is **not** certified for use in safety-critical maintenance decisions on real aircraft or industrial equipment.

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Contributing

Issues and pull requests are welcome.  
The most useful contributions right now are:

- Additional models that implement the common interface
- Better uncertainty estimation methods
- More thorough evaluation on FD002–FD004
- Improvements to the DocQA pipeline

---

**Built as a personal open-source project.**  
If it helps you learn or ship something useful, that’s the whole point.
