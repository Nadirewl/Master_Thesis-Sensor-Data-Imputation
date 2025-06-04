#Import Pypots Library
from pypots.optim import Adam
from pypots.imputation import GPVAE
from pypots.nn.functional import calc_mae,calc_mse,calc_rmse


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
from sklearn.model_selection import train_test_split
import optuna 
from optuna.visualization import plot_optimization_history
import os
import shutil




from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, Dataset
from pygrinder.missing_completely_at_random import mcar
from tqdm.auto import tqdm

import sensor_imputation_thesis.shared.load_data as load

torch.cuda.empty_cache()
#PatchTST might be an ideal choise if SAITS is too slow 

##Drop columns with different indexes while loading data.. Or the mean values 

#df=pd.read_parquet("/home/ec2-user/SageMaker/sensor-imputation-thesis/src/sensor_imputation_thesis/nadire/ny_df_for_pypots.parquet")

#use this df for comparison(remained the same with saits, vae, and other simplemodels )
df=pd.read_parquet("sensor-imputation-thesis/src/sensor_imputation_thesis/nadire/dffinal_for_comparison.parquet")
len(df)
#current length of the dataframe is 119439

# Check nan values in each column
for col in df.columns:
    print(f"Column {col} has {df[col].isna().sum()} NaN values")
    missing_rate=df[col].isna().sum()/len(df[col])
    print(f"Column {col} has {missing_rate} Missing_rate")


#Try with smaller dataset, size 4000
##SAMPLE the percengtage of the dataset, df.sample (averagely pick samples)
#not df.sample cuz it will randomly select 
original_size=len(df)
desired_fraction=0.3 #Select data every 3 minutes 
step=int(1/desired_fraction) #step_size=10 (sample every 10th (3/10) minute)

#Systematic sampling: Start at a random offset to avoid bias 
start=np.random.randint(0,step) #Random start between 0-9
df1=df.iloc[start::step].reset_index(drop=True)

print(f"Original size:{len(df)}, Sampled size: {len(df1)}")



# Custom Dataset class
class Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Data processing code
sensor_cols = [col for col in df1.columns if col != "time"]
data = df1[sensor_cols].values

#¤get feature names for printing mae later 
feature_names=df1[sensor_cols].columns.tolist()

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

#Apply time series split, here should be changed as well. 
#Split into train(60%), val(20%), and test (20%)
#train_data, temp_data=train_test_split(data_reshaped,test_size=0.4,shuffle=True) #
#val_data, test_data=train_test_split(temp_data, test_size=0.5, shuffle=False) # May 29th, this approach should not be used 


# test if this works 
train_size = int(0.6 * n_samples)
val_size = int(0.2 * n_samples)
test_size = n_samples - train_size - val_size

train_data = data_reshaped[:train_size]
val_data = data_reshaped[train_size:train_size + val_size]
test_data = data_reshaped[train_size + val_size:]

#normalize each feature independently
scalers={}

train_scaled = np.zeros_like(train_data)
val_scaled = np.zeros_like(val_data)
test_scaled = np.zeros_like(test_data)



for i in range(data_reshaped.shape[2]):
    scaler = MinMaxScaler(feature_range=(-1, 1)) #changed to -1,1
    # Flatten timesteps and samples for scaling
    train_scaled[:, :, i] = scaler.fit_transform(train_data[:, :, i].reshape(-1, 1)).reshape(train_data.shape[0], train_data.shape[1])
    val_scaled[:, :, i] = scaler.transform(val_data[:, :, i].reshape(-1, 1)).reshape(val_data.shape[0], val_data.shape[1])
    test_scaled[:, :, i] = scaler.transform(test_data[:, :, i].reshape(-1, 1)).reshape(test_data.shape[0], test_data.shape[1])
    scalers[i] = scaler  # Save scalers to inverse-transform later

#Inverse Scale
def inverse_scale(imputation, scalers):
    n_samples, n_timesteps, n_features = imputation.shape
    imputation_denorm = np.empty_like(imputation)
    
    for i in range(n_features):
        reshaped = imputation[:, :, i].reshape(-1, 1)
        inversed = scalers[i].inverse_transform(reshaped)
        imputation_denorm[:, :, i] = inversed.reshape(n_samples, n_timesteps)
    
    return imputation_denorm



#Optional: Artificially mask. Mask 20% of the data (MIT part). Try masking 30% here 
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


#?? Problem: Can't have the best input for testing
#1.Create synthetic test_data cuz if I drop nan values for test set, there's basically nothing left
#synthetic_data=np.random.randn(n_samples,n_steps,n_features)
#test_X_masked,test_mask=mcar_f(synthetic_data)
#test_X_ori=synthetic_data.copy() #Ground truth

# 2, Ensure no NaN values in synthetic data
#test_X_masked = np.nan_to_num(test_X_masked, nan=np.nanmean(test_X_masked))
#test_X_ori = np.nan_to_num(test_X_ori, nan=np.nanmean(test_X_ori))



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
use_mps = not args.no_mps and torch.backends.mps.is_available()

args.cuda = not args.no_cuda and torch.cuda.is_available()
print("CUDA available:", torch.cuda.is_available())


if args.cuda:
    device = torch.device("cuda")
    print("Using CUDA")
elif use_mps:
    device = torch.device("mps")
    print("Using MPS")
else:
    device = torch.device("cpu")
    print("Using CPU")

train_scaled = torch.tensor(train_scaled, dtype=torch.float32)
val_X_masked = torch.tensor(val_X_masked, dtype=torch.float32)
val_X_ori = torch.tensor(val_X_ori, dtype=torch.float32)

train_scaled = train_scaled.to(device)
val_X_masked = val_X_masked.to(device)
val_X_ori = val_X_ori.to(device)


#MLflow set up
mlflow.set_tracking_uri("http://localhost:5000")
client = mlflow.tracking.MlflowClient()
mlflow.set_experiment("GP_VAE_Final5.31")

saving_path="/home/ec2-user/SageMaker/sensor-imputation-thesis/src/sensor_imputation_thesis/nadire/gp_vae"
if os.path.exists(saving_path):
    shutil.rmtree(saving_path)
    os.makedirs(saving_path) 

# Optuna objective function
def objective(trial):
    params = {
        "lr": trial.suggest_float("lr", 1e-6, 1e-2, log=True),
        "epochs": trial.suggest_int("epochs", 10, 50),
        "batch_size": trial.suggest_int("batch_size", 32, 128, step=32),
        "length_scale": trial.suggest_float("length_scale",0.5,5.0),
        "beta": trial.suggest_float("beta",0.1,1.0)
 }

    with mlflow.start_run(run_name="GP-VAE-Final", nested=True) as run:
        mlflow.log_params(params)

        gp_vae = GPVAE(
            n_steps=data_reshaped.shape[1],
            n_features=data_reshaped.shape[2],
            latent_size=37, #should be the latent dimensions 
            encoder_sizes=(128,128), #should I change it here too?
            decoder_sizes=(256,256), #should I change the model size?
            kernel="cauchy",
            beta=params["beta"], #The weight of KL divergence in ELBO
            M=1,  #The number of Monte Carlo samples for ELBO estimation during training.
            K=1,  #The number of importance weights for IWAE model training loss.
            sigma=1.005, # The scale parameter for a kernel function
            length_scale=params["length_scale"], #The length scale parameter for a kernel function
            kernel_scales=1, #The number of different length scales over latent space dimensions
            window_size=24,  # Window size for the inference CNN.
            batch_size=params["batch_size"],
            # here we set epochs=10 for a quick demo, you can set it to 100 or more for better performance
            epochs=params["epochs"],
            # here we set patience=3 to early stop the training if the evaluting loss doesn't decrease for 3 epoches.
            # You can leave it to defualt as None to disable early stopping.
            patience=3,
            # give the optimizer. Different from torch.optim.Optimizer, you don't have to specify model's parameters when
            # initializing pypots.optim.Optimizer. You can also leave it to default. It will initilize an Adam optimizer with lr=0.001.
            optimizer=Adam(lr=params["lr"]),
            # this num_workers argument is for torch.utils.data.Dataloader. It's the number of subprocesses to use for data loading.
            # Leaving it to default as 0 means data loading will be in the main process, i.e. there won't be subprocesses.
            # You can increase it to >1 if you think your dataloading is a bottleneck to your model training speed
            num_workers=0,
            # just leave it to default as None, PyPOTS will automatically assign the best device for you.
            # Set it as 'cpu' if you don't have CUDA devices. You can also set it to 'cuda:0' or 'cuda:1' if you have multiple CUDA devices, even parallelly on ['cuda:0', 'cuda:1']
            device=device,
            # set the path for saving tensorboard and trained model files 
            saving_path="/home/ec2-user/SageMaker/sensor-imputation-thesis/src/sensor_imputation_thesis/nadire/gp_vae",
            # only save the best model after training finished.
            # You can also set it as "better" to save models performing better ever during training.
            model_saving_strategy="best",
        )



        # train the model on the training set, and validate it on the validating set to select the best model for testing in the next step
        gp_vae.fit(train_set={"X": train_scaled}, val_set={"X": val_X_masked, "X_ori": val_X_ori})
        gp_vae_results = gp_vae.predict({"X": test_X_masked}, n_sampling_times=2)
        gp_vae_imputation = gp_vae_results["imputation"]

        print(f"The shape of gp_vae_imputation is {gp_vae_imputation.shape}")

        # for error calculation, we need to take the mean value of the multiple samplings for each data sample
        mean_gp_vae_imputation = gp_vae_imputation.mean(axis=1)

        test_imputation_denorm = inverse_scale(mean_gp_vae_imputation, scalers)
        test_ori_denorm = inverse_scale(test_X_ori, scalers)


         # Calculate metrics
        mae_per_feature = []
        mse_per_feature=[]
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
    print("MSE per feature:", mse_per_feature)
    print("RMSE per feature",rmse_per_feature)
    print("Percentage MAE per feature:", percentage_mae_per_feature)
   

# Run Optuna study
mlflow.set_experiment("GP-VAE-Final")
with mlflow.start_run(run_name="GPVAE_Final)") as parent_run:
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)

    best_params = study.best_trial.params
    best_value = study.best_trial.value
    best_run_id = study.best_trial.user_attrs["mlflow_run_id"]

    # Log best parameters
    mlflow.log_params(best_params)

    # Log best metric(s)
    mlflow.log_metric("best_Avg_MAE", best_value)
    mlflow.log_param("best_run_id", best_run_id)

    print("Best Parameters:", best_params)
    print("Best Avg MAE:", best_value)

#clear directory again
saving_path="/home/ec2-user/SageMaker/sensor-imputation-thesis/src/sensor_imputation_thesis/nadire/gp_vae/Final_model"
if os.path.exists(saving_path):
    shutil.rmtree(saving_path)
    os.makedirs(saving_path) 


##Retrain with extended train data and test on the test 
with mlflow.start_run(run_name="Final_Pypots&val_for_all5.31", nested=True):
    GPVAE_best= GPVAE(
        n_steps=data_reshaped.shape[1],
        n_features=data_reshaped.shape[2],
        latent_size=37, #should be the latent dimensions 
        encoder_sizes=(128,128), #should I change it here too?
        decoder_sizes=(256,256), #should I change the model size?
        kernel="cauchy",
        beta=best_params["beta"], #The weight of KL divergence in ELBO
        M=1,  #The number of Monte Carlo samples for ELBO estimation during training.
        K=1,  #The number of importance weights for IWAE model training loss.
        sigma=1.005, # The scale parameter for a kernel function
        length_scale=best_params["length_scale"], #The length scale parameter for a kernel function
        kernel_scales=1, #The number of different length scales over latent space dimensions
        window_size=24,  # Window size for the inference CNN.
        batch_size=best_params["batch_size"],
        # here we set epochs=10 for a quick demo, you can set it to 100 or more for better performance
        epochs=best_params["epochs"],
        # here we set patience=3 to early stop the training if the evaluting loss doesn't decrease for 3 epoches.
        # You can leave it to defualt as None to disable early stopping.
        patience=3,
        # give the optimizer. Different from torch.optim.Optimizer, you don't have to specify model's parameters when
        # initializing pypots.optim.Optimizer. You can also leave it to default. It will initilize an Adam optimizer with lr=0.001.
        optimizer=Adam(lr=best_params["lr"]),
        # this num_workers argument is for torch.utils.data.Dataloader. It's the number of subprocesses to use for data loading.
        # Leaving it to default as 0 means data loading will be in the main process, i.e. there won't be subprocesses.
        # You can increase it to >1 if you think your dataloading is a bottleneck to your model training speed
        num_workers=0,
        # just leave it to default as None, PyPOTS will automatically assign the best device for you.
        # Set it as 'cpu' if you don't have CUDA devices. You can also set it to 'cuda:0' or 'cuda:1' if you have multiple CUDA devices, even parallelly on ['cuda:0', 'cuda:1']
        device=device,
        # set the path for saving tensorboard and trained model files 
        saving_path="/home/ec2-user/SageMaker/sensor-imputation-thesis/src/sensor_imputation_thesis/nadire/gp_vae/Final_model",
        # only save the best model after training finished.
        # You can also set it as "better" to save models performing better ever during training.
        model_saving_strategy="best",
    )


        ## Split train, val, test, sets again(90%,10%,10%)
        #concat train and val sets
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


    GPVAE_best.fit(train_set={"X": train_f}, val_set={"X": val_X_masked_2, "X_ori": val_X_ori_2})
    GPVAE_best_results =  GPVAE_best.predict({"X": test_X_masked}, n_sampling_times=2)
    gp_vae_imputation = GPVAE_best_results["imputation"]

    torch.save(GPVAE_best.model.state_dict(), "GPVAE_weights.pth")

    print(f"The shape of gp_vae_imputation is {gp_vae_imputation.shape}")

        # for error calculation, we need to take the mean value of the multiple samplings for each data sample
    mean_gp_vae_imputation = gp_vae_imputation.mean(axis=1)

    test_imputation_denorm = inverse_scale(mean_gp_vae_imputation, scalers)
    test_ori_denorm = inverse_scale(test_X_ori, scalers)


        # Calculate metrics
    mae_per_feature = []
    mse_per_feature=[]
    rmse_per_feature=[]
    percentage_mae_per_feature = []
    GPVAE_metrics_data=[]

    for i in range(n_features):
        imputation_i = test_imputation_denorm[:, :, i]
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

        GPVAE_metrics_data.append({
                        "feature": feature_names[i],
                        "masked_mae": mae_i,
                        "masked_mse": mse_i,
                        "masked_rmse": rmse_i,
                        "percentage_mae_vs_std": percentage_mae_i
                    })
        

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
        
    metrics_df = pd.DataFrame(GPVAE_metrics_data)
    print(metrics_df)
    metrics_df.to_parquet("/home/ec2-user/SageMaker/sensor-imputation-thesis/src/sensor_imputation_thesis/nadire/GPVAE_metrics.parquet", index=False)

    print("MAE per feature:", [float(x) for x in mae_per_feature])
    print("MSE per feature:", [float(x) for x in mse_per_feature])
    print("RMSE per feature",[float(x) for x in rmse_per_feature])








import pandas as pd
import numpy as np
import torch
from pypots.imputation import GPVAE







import sensor_imputation_thesis.shared.load_data as load

# === CONFIGURATION ===
original_data_path = "/home/ec2-user/SageMaker/sensor-imputation-thesis/src/sensor_imputation_thesis/nadire/dffinal_for_comparison.parquet"  
model_path = "sensor-imputation-thesis/src/sensor_imputation_thesis/nadire/gp_vae/Final_model/20250602_T013858/GPVAE.pypots"
output_path = "/home/ec2-user/SageMaker/sensor-imputation-thesis/src/sensor_imputation_thesis/nadire/imputed_wizGPVAE.parquet"  
n_steps = 20  # Window length

# === LOAD ORIGINAL DATA ===
df = pd.read_parquet(original_data_path)
time_col = df['time']
data = df.drop(columns=['time'])

# === PAD DATA IF NEEDED ===
n_features = data.shape[1]
padding_len = (n_steps - (len(data) % n_steps)) % n_steps
data_padded = np.pad(data.values, ((0, padding_len), (0, 0)), mode='constant', constant_values=np.nan)

# === RESHAPE INTO WINDOWS ===
data_reshaped = data_padded.reshape(-1, n_steps, n_features)

# === LOAD TRAINED MODEL ===
#model = GPVAE.load(model_path)  # as a positional argument

model = GPVAE.load_from_directory(model_path)



# === IMPUTE MISSING VALUES ===
results = model.predict({"X": data_reshaped}, n_sampling_times=2)
imputed = results["imputation"].mean(axis=1)

# === UNPAD AND RESTORE ORIGINAL SHAPE ===
imputed_flat = imputed.reshape(-1, n_features)[:len(data)]
imputed_df = pd.DataFrame(imputed_flat, columns=data.columns)
imputed_df['time'] = time_col.values

# === SAVE TO PARQUET ===
imputed_df.to_parquet(output_path, index=False)
print(f"✅ Imputed data saved to: {output_path}")
