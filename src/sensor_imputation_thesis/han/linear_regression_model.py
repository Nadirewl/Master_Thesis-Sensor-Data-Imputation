# create the dataframe
from data_insight import setup_duckdb
from duckdb import DuckDBPyConnection as DuckDB
import pandas as pd
from duckdb import DuckDBPyRelation as Relation
from pathlib import Path
import hashlib

pd.set_option('display.max_columns', None)

def load_engine_data(
    con: DuckDB, product_id: str, start: pd.Timestamp, stop: pd.Timestamp, tags: list[str]
) -> Relation:
    return con.sql(f"""
    SELECT {','.join(tags)}
    FROM timeseries
    WHERE
        time BETWEEN '{start}' AND '{stop}'
        AND pid = '{product_id}'
    """)

def get_tags_hash(tags):
    return hashlib.md5(','.join(tags).encode()).hexdigest()

start, stop = pd.Timestamp("2023-09-01"), pd.Timestamp("2024-09-01")
tags = [
        "time",
        "fr_eng",
        "fr_eng_ecs",
        "pr_cyl_max__6",
        "re_egb_area",
        "re_egb_pos",
        "te_exh_cyl_out__1",
        "te_air_ic_out__0",
        "pd_air_ic__0",
        "pr_baro",
        "pr_cyl_max__7",
    ]
product_id = "ced05645997a2804bde7aff0b6703b90"

cache = Path(f'/tmp/data_{get_tags_hash(tags)}_{product_id}.parquet')
if cache.exists():
    df_baseline = pd.read_parquet(cache)
else:
    con = setup_duckdb()
    df_baseline = load_engine_data(con, product_id, start, stop, tags).df()
    df_baseline.to_parquet(cache)

print(df_baseline.head(10))

# linear regression model with cross validation
import sklearn
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Choose a variable as y (e.g., 'fr_eng') and other variables as X
y = df_baseline['fr_eng']
X = df_baseline.drop(columns=['fr_eng', 'time'])

# Remove rows with NaN values 
X = X.dropna()
y = y[X.index]

# Alternatively, we can impute the missing values using a method such as the mean
# from sklearn.impute import SimpleImputer 
# imputer = SimpleImputer(strategy='mean') 
# X_imputed = imputer.fit_transform(X)

# Standardize the features
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# Create linear regression model
model = LinearRegression()

# Implement cross-validation (5-fold) and calculate performance metrics
cv_scores = cross_val_score(model, X_standardized, y, cv=5, scoring='r2')
mse_scores = cross_val_score(model, X_standardized, y, cv=5, scoring='neg_mean_squared_error')

# Fit the model on the entire dataset to report overall metrics
model.fit(X_standardized, y)
y_pred = model.predict(X_standardized)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)


# Output the model parameters (coefficients and intercept)
# Match the model's coefficients to their corresponding column names
coefficients = model.coef_
columns = X.columns
# Create a DataFrame for easy viewing 
coef_df = pd.DataFrame({'Column': columns, 'Coefficient': coefficients})
print(coef_df)
print("Model Intercept:", model.intercept_)

# Output the cross-validation scores and overall performance metrics
print("Cross-validation R^2 scores:", cv_scores)
print("Mean Cross-validation R^2 score:", np.mean(cv_scores))
print("Cross-validation MSE scores:", -mse_scores)  # Convert to positive since the score is negative
print("Mean Cross-validation MSE score:", -np.mean(mse_scores))
print("Overall MSE:", mse)
print("Overall R^2:", r2)