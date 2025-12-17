# Make a quick prediction

import joblib
import pandas as pd

# it loads the saved pipleline (preprocessor + regressor) and predicts Totla_Score for a sample student
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

# shows Predicted Total_Score: 87.13849999999998