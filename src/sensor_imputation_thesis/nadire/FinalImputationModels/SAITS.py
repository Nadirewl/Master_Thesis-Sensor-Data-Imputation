#Import Pypots Library
from pypots.optim import Adam
from pypots.imputation import SAITS
#from pypots.utils.metrics import calc_mae
from pypots.nn.functional import calc_mae
from pypots.nn.functional import calc_mse
from pypots.nn.functional import calc_rmse
import os
import shutil


import argparse
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.utils.data
import data_insight
from data_insight import setup_duckdb
from duckdb import DuckDBPyConnection as DuckDB
from duckdb import DuckDBPyRelation as Relation
from pathlib import Path
import hashlib
from duckdb import DuckDBPyConnection as DuckDB
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
import optuna 

from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, Dataset
from pygrinder.missing_completely_at_random import mcar
from tqdm.auto import tqdm

import sensor_imputation_thesis.shared.load_data as load

torch.cuda.empty_cache()
#PatchTST might be an ideal choise if SAITS is too slow 
#change the df input here for comparison
df=pd.read_parquet("/home/ec2-user/SageMaker/sensor-imputation-thesis/src/sensor_imputation_thesis/nadire/dffinal_for_comparison.parquet")
len(df)

# Check nan values in each column
#for col in df.columns:
 #   print(f"Column {col} has {df[col].isna().sum()} NaN values")
 #   missing_rate=df[col].isna().sum()/len(df[col])
 #   print(f"Column {col} has {missing_rate} Missing_rate")


#Try with smaller dataset, size 4000
##SAMPLE the percengtage of the dataset, df.sample (averagely pick samples)
original_size=len(df)
desired_fraction=0.3 #Select data every 3 minutes 
step=int(1/desired_fraction) #step_size=10 (sample every 10th (3/10) minute)

#Systematic sampling: Start at a random offset to avoid bias 
start=np.random.randint(0,step) #Random start between 0-9
df1=df.iloc[start::step].reset_index(drop=True)

#print(f"Original size:{len(df)}, Sampled size: {len(df1)}")



# Custom Dataset class
class Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Data processing code
sensor_cols = [col for col in df.columns if col != "time"]
data = df[sensor_cols].values

#¤get feature names for printing mae later 
feature_names=df[sensor_cols].columns.tolist()

## Convert data to 3D arrays of shape n_samples, n_timesteps, n_features, X_ori refers to the original data without missing values 
## Reconstruct all columns simultaneously  #num_features: 119
n_features = data.shape[1]  # exclude the time column
n_steps = 20 #60 (was 60 previously) #(TRY TO CHANGE HERE)  # # window length, 1440 steps = 24 hours of 1-minute data, but here is revised to 60 again
#total_elements = data.shape[0] * data.shape[1]
n_samples = data.shape[0] // n_steps 



# Reshape to (n_samples // n_steps, n_steps, n_features)
#data_reshaped = data.reshape((n_samples, n_steps, n_features))
data_reshaped=data[:n_samples*n_steps].reshape(n_samples,n_steps,n_features)
print(f"Reshaped data:{data_reshaped.shape}")

#Split into train, test, val, fit scaler only on the train set (prevent data leakage)

train_size = int(0.6 * n_samples)
val_size = int(0.2 * n_samples)
test_size = n_samples - train_size - val_size

train_data = data_reshaped[:train_size]
val_data = data_reshaped[train_size:train_size + val_size]
test_data = data_reshaped[train_size + val_size:]



##Normalization is important because of the nature of mse calculation of saits, columns with large 
#values dominate the loss, making metrics meaningless. SAITS computes MSE/MAE column-wise and averages 
#them across all columns 
#  Apply minmax scaler here 
#normalize each feature independently
scalers={}



train_scaled = np.zeros_like(train_data)
val_scaled = np.zeros_like(val_data)
test_scaled = np.zeros_like(test_data)



for i in range(data_reshaped.shape[2]):
    scaler = MinMaxScaler(feature_range=(-1, 1)) 
    # Flatten timesteps and samples for scaling
    train_scaled[:, :, i] = scaler.fit_transform(train_data[:, :, i].reshape(-1, 1)).reshape(train_data.shape[0], train_data.shape[1])
    val_scaled[:, :, i] = scaler.transform(val_data[:, :, i].reshape(-1, 1)).reshape(val_data.shape[0], val_data.shape[1])
    test_scaled[:, :, i] = scaler.transform(test_data[:, :, i].reshape(-1, 1)).reshape(test_data.shape[0], test_data.shape[1])
    scalers[i] = scaler  # Save scalers to inverse-transform later

#Inverse Scale
def inverse_scale(imputation, scalers):
    n_features = imputation.shape[2]
    imputation_denorm = np.empty_like(imputation)
    
    for i in range(n_features):
        imputation_denorm[:, :, i] = scalers[i].inverse_transform(imputation[:, :, i].reshape(-1, 1)).reshape(imputation.shape[0], imputation.shape[1])
    
    return imputation_denorm  


# Artificially mask. Mask 30% to compare with GP-VAE
def mcar_f(X, mask_ratio=0.3):
    """Apply MCAR only to observed values."""
    observed_mask=~np.isnan(X) #find observed positions
    artificial_mask=mcar(X,mask_ratio).astype(bool) #generate MCAR mask, cast to boolean
    #combine masks 
    combined_mask=observed_mask & artificial_mask

    #Apply masking
    X_masked=X.copy()
    X_masked[combined_mask]=np.nan
    return X_masked,combined_mask



#Use mcar on validation data 
val_X_masked, val_mask =mcar_f(val_scaled)
val_X_ori=val_scaled.copy() 

test_X_masked, test_mask =mcar_f(test_scaled)
test_X_ori=test_scaled.copy() 


class Config:
    no_cuda = False
    no_mps = False
    seed = 1

args=Config()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)


args.cuda = not args.no_cuda and torch.cuda.is_available()
print("CUDA available:", torch.cuda.is_available())
use_mps = not args.no_mps and torch.backends.mps.is_available()


if args.cuda:
    device = torch.device("cuda")
    print("Using CUDA")
elif use_mps:
    device = torch.device("mps")
    print("Using MPS")
else:
    device = torch.device("cpu")
    print("Using CPU")

train_scaled = torch.tensor(train_scaled, dtype=torch.float32).to(device)
val_X_masked = torch.tensor(val_X_masked, dtype=torch.float32).to(device)
val_X_ori = torch.tensor(val_X_ori, dtype=torch.float32).to(device)


#MLflow set up
mlflow.set_tracking_uri("http://localhost:5000")
client = mlflow.tracking.MlflowClient()


#clear directory before training 
# Define the directory to be cleared
saving_path = '/home/ec2-user/SageMaker/sensor-imputation-thesis/src/sensor_imputation_thesis/nadire/SAITS'

# Clear the directory by deleting all its contents
if os.path.exists(saving_path):
    shutil.rmtree(saving_path)
    os.makedirs(saving_path)  # Recreate the directory after clearing


# Optuna objective function
def objective(trial):
    params = {
        "n_layers": trial.suggest_int("n_layers", 2, 4),
        "lr": trial.suggest_float("lr", 1e-6, 1e-2, log=True),  
        "epochs": trial.suggest_int("epochs", 10, 20),
        "batch_size": trial.suggest_int("batch_size", 4, 16)
    }

    with mlflow.start_run(run_name="SAITS_finaltocompare", nested=True) as run:
        mlflow.log_params(params)

        saits = SAITS(
            n_steps=data_reshaped.shape[1],
            n_features=data_reshaped.shape[2],
            n_layers=params["n_layers"],
            d_model=512,
            optimizer=Adam(lr=params["lr"]),
            ORT_weight=1.0,
            MIT_weight=1.0,
            batch_size=params["batch_size"],
            epochs=params["epochs"],
            d_ffn=512,
            n_heads=8,
            d_k=64,
            d_v=64,
            dropout=0.1,
            attn_dropout=0.1,
            diagonal_attention_mask=True,
            patience=6,
            num_workers=0,
            device=device,
            #saving_path="/home/ec2-user/SageMaker/sensor-imputation-thesis/src/sensor_imputation_thesis/nadire/SAITS",
            #model_saving_strategy="best",
        )

        saits.fit(train_set={"X": train_scaled}, val_set={"X": val_X_masked, "X_ori": val_X_ori})
        test_imputation = saits.predict({"X": test_X_masked})["imputation"]
        test_imputation_denorm = inverse_scale(test_imputation, scalers)
        test_ori_denorm = inverse_scale(test_X_ori, scalers)
        

        # Calculate metrics, adding mse
        mae_per_feature = []
        mse_per_feature= []
        rmse_per_feature=[]
        percentage_mae_per_feature = []

        for i in range(n_features):
            imputation_i = test_imputation_denorm[:, :, i]
            ground_truth_i = test_ori_denorm[:, :, i]
            mask_i = test_mask[:, :, i]
            if np.isnan(imputation_i).any() or np.isnan(ground_truth_i).any():
                continue
            mae_i = calc_mae(imputation_i, ground_truth_i, mask_i)
            mae_per_feature.append(mae_i)
            mse_i = calc_mse(imputation_i, ground_truth_i, mask_i)
            mse_per_feature.append(mse_i)
            rmse_i = calc_rmse(imputation_i, ground_truth_i, mask_i)
            rmse_per_feature.append(rmse_i)

            #Calculate the original standard deviation for the feature
            std_dev_i = np.std(ground_truth_i[mask_i == 1])
             # Calculate the percentage of MAE relative to the standard deviation   
            if std_dev_i != 0:
                percentage_mae_i = (mae_i / std_dev_i) * 100
                percentage_mae_per_feature.append(percentage_mae_i)
            else:
                 percentage_mae_i = float('inf')


            mlflow.log_metric(f"MAE_{feature_names[i]}", mae_i)
            mlflow.log_metric(f"MSE_{feature_names[i]}",mse_i)
            mlflow.log_metric(f"RMSE_{feature_names[i]}",rmse_i)
            mlflow.log_metric(f"Percentage_MAE_{feature_names[i]}", percentage_mae_i)

        avg_mae = np.mean(mae_per_feature)
        avg_mse=np.mean(mse_per_feature)
        avg_rmse=np.mean(rmse_per_feature)
       
        mlflow.log_metric("avg_mae", avg_mae)
        mlflow.log_metric("avg_mse",avg_mse)
        mlflow.log_metric("avg_rmse", avg_rmse)

        trial.set_user_attr("mlflow_run_id", run.info.run_id)

        return avg_mae

    print("MAE per feature:", mae_per_feature)
    print("MSE per feature",mse_per_feature)
    print("RMSE per feature",rmse_per_feature)
    print("Percentage MAE per feature:", percentage_mae_per_feature)
   

# Run Optuna study
mlflow.set_experiment("SAITS-Final")
with mlflow.start_run(run_name="SAITS_Optuna_Study") as parent_run:
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)

    best_params = study.best_trial.params
    best_value = study.best_trial.value
    best_run_id = study.best_trial.user_attrs["mlflow_run_id"]

    # Log best parameters
    mlflow.log_params(best_params)

    # Log best metric(s)
    mlflow.log_metric("best_Average_MAE", best_value)
    mlflow.log_param("best_run_id", best_run_id)

print("Best Parameters:", best_params)
print("Best Avg MAE:", best_value)

#clear directory before training 
# Define the directory to be cleared
saving_path = "/home/ec2-user/SageMaker/sensor-imputation-thesis/src/sensor_imputation_thesis/nadire/SAITS_Final"

# Clear the directory by deleting all its contents
if os.path.exists(saving_path):
    shutil.rmtree(saving_path)
    os.makedirs(saving_path)  # Recreate the directory after clearing


# Re-run the model with best parameters (to be revised, combine train and val, and then use 80% of train and val combines as train, 20% as val, and the test remain the same 
with mlflow.start_run(run_name="Final_SAITS&5.31_val_for_all", nested=True):
    saits_best = SAITS(
        n_steps=data_reshaped.shape[1],
        n_features=data_reshaped.shape[2],
        n_layers=4,
        d_model=512,
        optimizer=Adam(lr=0.0039),
        ORT_weight=1.0,
        MIT_weight=1.0,
        batch_size=best_params["batch_size"],
        epochs=best_params["epochs"],
        d_ffn=512,
        n_heads=8,
        d_k=64,
        d_v=64,
        dropout=0.1,
        attn_dropout=0.1,
        diagonal_attention_mask=True,
        patience=6,
        num_workers=0,
        device=device,
        saving_path="/home/ec2-user/SageMaker/sensor-imputation-thesis/src/sensor_imputation_thesis/nadire/SAITS_Final", 
    )

    ## Split train, val, test, sets again(90%,10%,10%)
    val_scaled = torch.from_numpy(val_scaled).float().to(device)
    combined_train = torch.cat([train_scaled, val_scaled], dim=0) #use val_scaled data that was before masking 
    #split into 90% and 10%
    split_index = int(combined_train.shape[0] * 0.9)

    train_f = combined_train[:split_index]
    val_f = combined_train[split_index:]

    val_f_cpu = val_f.cpu().numpy()  # if it's a PyTorch tensor
    val_X_masked_2, val_mask = mcar_f(val_f_cpu)
    #already in the tensor and on the device
    val_X_ori_2=val_f.clone()
    
    #Move to same device 
    device = train_f.device
    val_X_masked_2 = torch.from_numpy(val_X_masked_2).float()
    val_mask=torch.from_numpy(val_mask).float()
    val_X_masked_2 = val_X_masked_2.to(device)
    val_mask=val_mask.to(device)
    val_X_ori_2 = val_X_ori_2.to(device)



    #test set remained the same

    saits_best.fit(train_set={"X": train_f}, val_set={"X": val_X_masked_2, "X_ori": val_X_ori_2})
    test_imputation_best = saits_best.predict({"X": test_X_masked})["imputation"]
    test_imputation_best_denorm = inverse_scale(test_imputation_best, scalers)
    test_ori_denorm = inverse_scale(test_X_ori, scalers)

   #Load Model
    torch.save(saits_best.model.state_dict(), "saits_weights.pth")

    
    # Calculate metrics, adding mse
    mae_per_feature = []
    mse_per_feature= []
    rmse_per_feature=[]
    percentage_mae_per_feature = []
    SAITS_metrics_data=[]


    for i in range(n_features):
        imputation_i = test_imputation_best_denorm[:, :, i]
        ground_truth_i = test_ori_denorm[:, :, i]
        mask_i = test_mask[:, :, i]  # 1 = missing value (i.e., imputed), 0 = originally present

        # Only compute on the masked (i.e., artificially missing) and valid positions
        valid_mask = (mask_i == 1) & ~np.isnan(ground_truth_i) & ~np.isnan(imputation_i)
        
        if np.sum(valid_mask) == 0:
            print(f"No valid masked values for feature {feature_names[i]}. Skipping.")
            continue

        y_true = ground_truth_i[valid_mask]
        y_pred = imputation_i[valid_mask]

        mae_i = calc_mae(y_pred, y_true, np.ones_like(y_true))  # already filtered, so mask is 1s
        mae_per_feature.append(mae_i)

        mse_i=calc_mse(y_pred, y_true, np.ones_like(y_true)) 
        mse_per_feature.append(mse_i)

        rmse_i = np.sqrt(mean_squared_error(y_true, y_pred))
        rmse_per_feature.append(rmse_i)

        std_dev_i = np.std(y_true)
        percentage_mae_i = (mae_i / std_dev_i * 100) if std_dev_i != 0 else float('inf')
        percentage_mae_per_feature.append(percentage_mae_i)

        SAITS_metrics_data.append({
                        "feature": feature_names[i],
                        "masked_mae": mae_i,
                        "masked_mse": mse_i,
                        "masked_rmse": rmse_i,
                        "percentage_mae_vs_std": percentage_mae_i
                    })
            
    
        # Logging
        mlflow.log_metric(f"MAE_{feature_names[i]}", mae_i)
        mlflow.log_metric(f"MSE_{feature_names[i]}",mse_i)
        mlflow.log_metric(f"RMSE_{feature_names[i]}", rmse_i)
        mlflow.log_metric(f"Percentage_MAE_{feature_names[i]}", percentage_mae_i)

    #save result as df
    metrics_df = pd.DataFrame(SAITS_metrics_data)
    print(metrics_df)
    metrics_df.to_parquet("/home/ec2-user/SageMaker/sensor-imputation-thesis/src/sensor_imputation_thesis/nadire/SAITS_metrics.parquet", index=False)

    avg_mae = np.mean(mae_per_feature)
    avg_mse=np.mean(mse_per_feature)
    avg_rmse=np.mean(rmse_per_feature)

    mlflow.log_metric("avg_mae", avg_mae)
    mlflow.log_metric("avg_mse",avg_mse)
    mlflow.log_metric("avg_rmse", avg_rmse)

print("length of mae list:",len(mae_per_feature))
print(f"✅ {feature_names[i]}: MAE={mae_i:.4f}, RMSE={rmse_i:.4f}, %MAE/STD={percentage_mae_i:.2f}%")
print("MAE per feature:", [float(x) for x in mae_per_feature])
print("MSE per feature:", [float(x) for x in mse_per_feature])
print("RMSE per feature",[float(x) for x in rmse_per_feature])



#Imputation
import torch
import pandas as pd
import numpy as np
from pypots.imputation import SAITS

# === Load full dataset ===
df = pd.read_parquet("/home/ec2-user/SageMaker/sensor-imputation-thesis/src/sensor_imputation_thesis/nadire/dffinal_for_comparison.parquet")

# === Drop time column for model input ===
data = df.drop(columns=['time']).values

# === Reshape and pad ===
#if added the fraction selection, here should add the preprocessing step as well 
n_steps = 20
n_features = data.shape[1]
padding_len = (n_steps - (len(data) % n_steps)) % n_steps
data_padded = np.pad(data, ((0, padding_len), (0, 0)), mode='constant', constant_values=np.nan)
data_reshaped = data_padded.reshape(-1, n_steps, n_features)

# === Initialize model ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SAITS(
    n_steps=n_steps,
    n_features=n_features,
    n_layers=4,
    d_model=512,
    d_ffn=512,
    n_heads=8,
    d_k=64,
    d_v=64,
    dropout=0.1,
    attn_dropout=0.1,
    diagonal_attention_mask=True,
    epochs=1,
    patience=1,
    device=device
)

# === Load saved weights ===
model.model.load_state_dict(torch.load(
    "/home/ec2-user/SageMaker/sensor-imputation-thesis/src/sensor_imputation_thesis/nadire/saits_weights.pth",
    map_location=device
))

# === Impute ===
results = model.predict({"X": data_reshaped})
imputed = results["imputation"].mean(axis=1)
imputed_flat = imputed.reshape(-1, n_features)[:len(data)]  # Trim padding

# === Create DataFrame with imputed values and reattach time column ===
imputed_df = pd.DataFrame(imputed_flat, columns=df.columns.drop('time'))
imputed_df['time'] = df['time'].iloc[:len(imputed_df)].values
imputed_df = imputed_df[['time'] + list(imputed_df.columns.drop('time'))]  # Reorder

# === Save the final imputed dataset ===
imputed_df.to_parquet("/home/ec2-user/SageMaker/sensor-imputation-thesis/src/sensor_imputation_thesis/nadire/saitsimputed.parquet", index=False)

print("✅ Final imputation completed and saved (with time column).")
