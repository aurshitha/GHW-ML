import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def preprocess_data(df, target_cols):
    # Remove target column(s) to avoid data leakage
    X = df.drop(columns=target_cols)

    # Automatically detect feature types
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    # Numeric preprocessing:
    # - Fill missing values with median (robust to outliers)
    # - Scale features to zero mean and unit variance
    numeric_pipeline = SimpleImputer(strategy='median')

     # Categorical preprocessing:
    # - Fill missing values with most frequent category
    # - Convert categories to numerical form using one-hot encoding
    categorical_pipeline = OneHotEncoder(handle_unknown='ignore')

   # Combine numeric and categorical preprocessing in one transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', categorical_pipeline, categorical_features),
        ]
    )
    return preprocessor, X

''' This function builds a reusable preprocessing pipeline that handles
 missing values, scales numeric features, encodes categorical features,
 and prevents data leakage before model training.
'''