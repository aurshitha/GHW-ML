# GHWMLRepo
Global Hack Week AI/ML (December 2025) Repository for creating a ML model from scratch. 

[Dataset Link](https://www.kaggle.com/datasets/mahmoudelhemaly/students-grading-dataset)


**Project Title**: Student Performance — Regression & EDA (GHW AI & ML 2025)

**Project Description**: This repository contains code and notebooks to explore student performance data, train a regression model that predicts a student's average (Total_Score) from academic and behavioral features, and demonstrate making predictions with the saved model pipeline.

**Quick Start**
- **Python**: This project targets Python 3.8+.
- **Create venv and install deps**: open PowerShell at the repo root and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

- **Run exploratory notebook**: open `notebooks/exploration.ipynb` in VS Code or Jupyter and run all cells to view plots and analysis.

- **Train regression model**: the repo includes `src/train_regression.py`. To train and save the pipeline run:

```powershell
python src\train_regression.py
```

This prints MSE and R^2 and saves the pipeline to `models/regression_model.joblib`.

- **Predict with saved model**: use the example script or the snippet below to load `models/regression_model.joblib` and call `predict()` on a pandas DataFrame of raw features.

**Repository Structure**
- **`data/`**: raw dataset files (CSV/JSON) used by scripts and the notebook.
- **`notebooks/`**: `exploration.ipynb` — EDA, plots, and preprocessing experiments.
- **`src/`**: core Python scripts
  - `train_regression.py` — trains a scikit-learn pipeline (preprocessor + RandomForestRegressor) and saves it to `models/`.
  - `preprocess.py` — builds the `ColumnTransformer` preprocessor used by the pipeline.
  - `utils.py` — data loading helpers (robust to filename variants in `data/`).
  - `evaluate.py` — (placeholder / optional) for additional evaluation logic.
- **`models/`**: saved model artifacts (`regression_model.joblib`, and optionally `preprocessor.joblib`).
- **`app/`**: (empty) intended place for a small web demo if you want to add one.

**Data and Expected Columns**
- Scripts expect the following columns by default (in `data/` CSV):
  - `Attendance (%)`
  - `Midterm_Score`
  - `Final_Score`
  - `Projects_Score`
  - `Study_Hours_per_Week`

If your CSV uses different column names, either rename columns or update `src/utils.py` and `src/preprocess.py` accordingly.

**Example: Prediction Snippet**
Save this as `predict_example.py` at the repo root or run interactively in a notebook.

```python
import pandas as pd
import joblib

model = joblib.load("models/regression_model.joblib")

sample = pd.DataFrame([{
    'Attendance (%)': 85,
    'Midterm_Score': 80,
    'Final_Score': 90,
    'Projects_Score': 88,
    'Study_Hours_per_Week': 10
}])

pred = model.predict(sample)
print("Predicted Total_Score:", pred[0])
```

**Troubleshooting**
- If `FileNotFoundError` occurs when running training: confirm the CSV is present in `data/` — common filenames in this repo include `Students Performance Dataset.csv` (with spaces) or `Students_Performance_Dataset.csv` (underscore). `src/utils.py` contains a fallback search but confirm the file exists.
- If `KeyError` arises for missing columns: open the CSV and verify header names match the expected list above.
- If plotting in the notebook fails: ensure `matplotlib`/`seaborn` are installed and the kernel is using the same environment where packages were installed.

**Next Enhancements (ideas)**
- Add a small API (Flask / FastAPI / Streamlit) in `app/` to serve predictions.
- Add `src/train_classification.py` to train a classification target (e.g., `Pass/Fail`).
- Add unit tests for data loading and the prediction pipeline.

**Contributing**
- Feel free to open an issue or PR. Keep changes small and provide a short description of why the change is needed.

<!---**License & Contact**
- Add your preferred license file (e.g., `LICENSE`) if you will share this publicly.
- For questions, contact the project owner (add your email here). -->