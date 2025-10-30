
## Wine Quality Prediction – MLflow + DVC Demo


A clean, professional, and fully reproducible machine learning project that predicts whether a red wine is *high quality* (rating ≥ 7) using its chemical properties.

Built with:
- **MLflow** – full experiment tracking, metrics, and model registry
- **DVC** – data version control and reproducible pipelines
- **scikit-learn** – Random Forest classifier
- **Poetry** – dependency and environment management

Perfect for learning MLOps, showcasing in a portfolio, or using as a template.

---

### Dataset

**Red Wine Quality** – [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality)  
- **1,599 samples**  
- **11 features** (fixed acidity, citric acid, alcohol, etc.)  
- **Target**: `high_quality` = 1 if quality ≥ 7

> Small, real, and scientifically meaningful — ideal for demos.

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/alihojatnia/Wine.git
cd Wine

# 2. Install dependencies
poetry install
poetry shell

# 3. Download the raw dataset
mkdir -p data/raw
curl -L -o data/raw/winequality-red.csv \
  https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv

# 4. Track data with DVC
dvc add data/raw/winequality-red.csv
git add data/raw/winequality-red.csv.dvc
git commit -m "Add raw wine data with DVC"

# 5. Run the full pipeline
dvc repro
```

---

## View Experiments

```bash
mlflow ui
```

Open [http://localhost:5000](http://localhost:5000) to explore:
- Hyperparameters
- Accuracy & F1 score
- Model artifacts
- Registered models (auto-registered if F1 > 0.70)

---

## Project Structure

```text
├── data/
│   ├── raw/          # DVC-tracked raw CSV
│   └── processed/    # train/test split (generated)
├── src/
│   ├── data.py       # load, clean, split
│   └── model.py      # train, evaluate, log with MLflow
├── models/           # saved model (rf.pkl)
├── metrics/          # JSON metrics (DVC-tracked)
├── params.yaml       # pipeline parameters
├── dvc.yaml          # DVC pipeline definition
├── pyproject.toml    # Poetry dependencies
└── README.md         
```

---

## Reproducibility Guaranteed

- **Data** → versioned with DVC (`*.dvc` files)  
- **Pipeline** → `dvc repro` rebuilds everything from scratch  
- **Environment** → locked via `poetry.lock`  
- **Experiments** → tracked in `mlruns/` + MLflow UI  

> Change a parameter → commit → `dvc repro` → new version, new run. **Zero surprises.**

---

## Try It Yourself

Edit `params.yaml`:

```yaml
train:
  max_depth: 12      # try deeper trees
  n_estimators: 300
```

Then:

```bash
git commit -am "Experiment: deeper forest"
dvc repro
```

New model, new metrics, new MLflow run — all versioned.

---


## Contributing

Ideas? Improvements? PRs welcome!  
Try:
- Adding XGBoost or Logistic Regression
- Feature engineering (e.g., alcohol × pH)
- Unit tests or GitHub Actions CI

---

## License

[MIT License](LICENSE) – free to use, modify, and share.

