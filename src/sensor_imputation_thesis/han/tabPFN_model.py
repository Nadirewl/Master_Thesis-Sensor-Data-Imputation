import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNRegressor 

# This dataset is from engine_id (product_id = "ced05645997a2804bde7aff0b6703b90") and 2023-11-01 - 2024-10-31
# maximum number of samples 10000 officially supported by TabPFN
path = "/home/ec2-user/SageMaker/sensor-imputation-thesis/src/sensor_imputation_thesis/han/dataframe_forxgboost"
df = pd.read_parquet(path)

# Filter the DataFrame as the engine is running
filtered_df = df[(df['fr_eng'] > (10/60)) & (df['fr_eng_ecs'] > (10/60))]

# Choose a variable as y ('pr_baro') and other variables as X
# Here we drop the nans since tabpfn could not deal with missing values
filtered_df = filtered_df.iloc[:, filtered_df.nunique().values > 1]
filtered_df = filtered_df.iloc[:, filtered_df.isna().mean().values < 0.95]

df_length = len(filtered_df)
train_windows = filtered_df[:int(df_length * 0.8)]
test_windows = filtered_df[int(df_length * 0.8):]

# Sampling
train_sampled = train_windows.sample(frac=0.1, random_state=42)
test_sampled = test_windows.sample(frac=0.1, random_state=42)

print(len(train_sampled))
print(len(test_sampled))

x_columns = ['pr_cyl_max_mv_iso', 'fr_eng_setpoint', 'pr_cyl_max__0', 'te_exh_cyl_out__0', 'pr_cyl_comp__0', 'pr_air_scav_ecs', 'pd_air_ic__0', 'te_air_scav_rec', 'te_air_ic_out__0_2', 'pr_cyl_comp_mv_iso', 'se_mip__0', 'te_air_ic_out__0', 'pr_air_scav', 'fr_eng_ecs', 'pr_air_scav_iso', 'fr_eng']
X_train = train_sampled[x_columns]
y_train = train_sampled['pr_baro']

X_test = test_sampled[x_columns]
y_test = test_sampled['pr_baro']

# Initialize the regressor
regressor = TabPFNRegressor()
regressor.fit(X_train, y_train)

# Predict on the test set
predictions = regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
rmse = np.sqrt(mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R² Score:", r2)