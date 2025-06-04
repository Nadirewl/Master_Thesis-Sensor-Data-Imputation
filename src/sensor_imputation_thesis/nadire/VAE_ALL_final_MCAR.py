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
from optuna.visualization import plot_optimization_history


from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset,DataLoader, Dataset
from tqdm.auto import tqdm

import sensor_imputation_thesis.shared.load_data as load

from typing import Any


#Flow

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# set tracking_uri:
mlflow.set_tracking_uri("http://localhost:5000")

#Load data 
df=pd.read_parquet("/home/ec2-user/SageMaker/sensor-imputation-thesis/src/sensor_imputation_thesis/nadire/dffinal_for_comparison.parquet")

# Check nan values in each column
for col in df.columns:
    print(f"Column {col} has {df[col].isna().sum()} NaN values")



#check the length of the df1 and the length of df to check how many cols rows are dropped 
print("shape of original df:")
print("Datasize:",len(df))
print("Colnumber:",len(df.columns))

df1=df.dropna(axis=1,how='all')
df1=df1.dropna(axis=0,how='any')


# Select numeric columns
numeric_df = df1.select_dtypes(include="number")
print(numeric_df)

data=numeric_df

#split dataset
train_size = int(0.6 * len(data))
val_size=int(0.2 * len(data))

train_df = data.iloc[:train_size]
val_df = data.iloc[train_size:train_size + val_size]
test_df= data.iloc[train_size + val_size:]

#revised functions, because masking will be done first, and then scaling. 
def preprocess(train_df, val_df, test_df, placeholder=-2):
    data_min = {}
    data_max = {}
    for col in train_df.columns:
        observed_values = train_df[col][train_df[col] != placeholder]
        data_min[col] = observed_values.min()
        data_max[col] = observed_values.max()

    def scale_df(df):
        scaled_df = df.copy()
        for col in df.columns:
            min_val = data_min[col]
            max_val = data_max[col]
            scale_range = max_val - min_val if max_val != min_val else 1e-8
            scaled_col = (df[col] - min_val) / scale_range * 2 - 1
            scaled_col[df[col] == placeholder] = placeholder
            scaled_df[col] = scaled_col
        return scaled_df

    return scale_df(train_df), scale_df(val_df), scale_df(test_df), data_min, data_max


def mask_f(df, missing_rate=0.3, seed=42, placeholder=-2):
    torch.manual_seed(seed)
    np.random.seed(seed)

    artificial_mask = pd.DataFrame(
        np.random.rand(*df.shape) > missing_rate,
        columns=df.columns,
        index=df.index
    )

    df_masked = df.where(artificial_mask, placeholder)

    return df_masked, artificial_mask




def inverse_scale_tensor(tensor, data_min_tensor, data_max_tensor, is_input=False, placeholder=-2.0):
    """
    Inversely scales a tensor from [-1, 1] back to original range using min/max.
    Args:
        tensor (torch.Tensor): Scaled tensor.
        data_min_tensor (torch.Tensor): Min values per feature.
        data_max_tensor (torch.Tensor): Max values per feature.
        is_input (bool): If True, reapply placeholder mask.
        placeholder (float): Placeholder value used for missing data.
    Returns:
        torch.Tensor: Unscaled tensor.
    """
    scale_range = data_max_tensor - data_min_tensor
    scale_range[scale_range == 0] = 1e-8  # Avoid division by zero

    # Reshape for broadcasting if needed
    if tensor.dim() == 2 and data_min_tensor.dim() == 1:
        data_min_tensor = data_min_tensor.unsqueeze(0)
        data_max_tensor = data_max_tensor.unsqueeze(0)
        scale_range = scale_range.unsqueeze(0)

    unscaled = (tensor + 1) / 2 * scale_range + data_min_tensor

    if is_input:
        # Reapply placeholder mask
        placeholder_mask = (tensor == placeholder)
        unscaled[placeholder_mask] = float('nan')

    return unscaled


class VAE(nn.Module):
    def __init__(self, input_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.tanh(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    
    def forward(self, x, mask=None):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z) 
        return recon, mu, logvar



# Reconstruction + KL divergence losses summed over target column and batch
def loss_function(recon_x, x, mu, logvar):
    # MSE only on masked data for imputation task 
    valid_mask=(x==-2)
    recon_loss=torch.mean(valid_mask*(x-recon_x)**2)  #mse 
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Combine losses using lambda
    loss = recon_loss+KLD  #adding kld weight? 
    return loss


def train(epoch, model:VAE, train_loader, optimizer, device, args):
    model.train()
    train_loss = 0
    avg_train_loss = 0  # Initialize average_train_loss
    pbar = tqdm(total=len(train_loader), desc="Training")
    for batch_idx, (data, mask) in enumerate(train_loader):
        data, mask = data.to(device), mask.to(device)
        optimizer.zero_grad()

        recon_batch, mu, logvar = model(data)
        if torch.isnan(recon_batch).any():
            print("NaN or in recon_batch!")
            break
        loss = loss_function(recon_batch, data, mu, logvar)
        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError(f"NaN or Inf detected in loss at batch {batch_idx}. Check scaling or model outputs!")
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) #add gradient clipping
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            pbar.set_description(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item() / len(data),
                )
            )

        pbar.update(1)
    pbar.close()
                                                                                        
    avg_train_loss = train_loss / len(train_loader.dataset)  # Update after loop
    print("====> Epoch: {} Average train loss: {:.4f}".format(epoch, avg_train_loss))
    return avg_train_loss


feature_names=data.columns.tolist()

def test(epoch, model, data_min, data_max, test_loader, original, device):
    model.eval()
    test_loss = 0

    # Separate metrics for masked and unmasked
    mae_total = {}
    mse_total = {}
    rmse_total = {}
    total_samples = {}


    with torch.no_grad():
        for batch_idx, (data, mask) in enumerate(test_loader):
            data, mask = data.to(device), mask.to(device)
            recon_batch, mu, logvar = model(data)
            original_data = original[batch_idx * data.size(0) : (batch_idx + 1) * data.size(0)]
            original_data=original_data.to(device)

            loss = loss_function(recon_batch, data, mu, logvar)
            test_loss += loss.item()

            data_min_tensor = torch.tensor([data_min[col] for col in feature_names], dtype=torch.float32).to(device)
            data_max_tensor = torch.tensor([data_max[col] for col in feature_names], dtype=torch.float32).to(device)

            recon_unscaled = inverse_scale_tensor(recon_batch, data_min_tensor, data_max_tensor)

            for col in range(data.shape[1]):
                recon_col = recon_unscaled[:, col]

                # Masked (missing) positions
                col_missing_mask = (mask[:, col] == 0)
                data_original_col=original_data[:,col] #revision
                diff=recon_col-data_original_col
                valid_mask = ~torch.isnan(diff) & col_missing_mask
                if valid_mask.sum() > 0:
                    valid_diff = diff[valid_mask]
                    mae_total[col] = mae_total.get(col, 0) + torch.sum(torch.abs(valid_diff)).item()
                    mse_total[col] = mse_total.get(col, 0) + torch.sum(valid_diff ** 2).item()
                    rmse_total[col] = rmse_total.get(col, 0) + torch.sqrt(torch.sum(valid_diff ** 2)).item()
                    total_samples[col] = total_samples.get(col, 0) + valid_mask.sum().item()

    # Compute averages
    avg_test_loss = test_loss / len(test_loader.dataset)
    avg_mae = {col: mae_total[col] / total_samples[col] for col in mae_total}
    avg_mse = {col: mse_total[col] / total_samples[col] for col in mse_total}
    avg_rmse = {col: rmse_total[col] / total_samples[col] for col in rmse_total}

    avg_mae_value = np.mean(list(avg_mae.values())) if avg_mae else float('nan')
    avg_mse_value = np.mean(list(avg_mse.values())) if avg_mse else float('nan')
    avg_rmse_value = np.mean(list(avg_rmse.values())) if avg_rmse else float('nan')

    # Print results
    print(f"====> Epoch: {epoch} Test Loss: {avg_test_loss:.4f}")
    if avg_mae:
        for col in avg_mae:
            col_name = feature_names[col] if col < len(feature_names) else f"col_{col}"
            print(f"====> Column {col_name} [Masked]   - MAE: {avg_mae[col]:.4f}, MSE: {avg_mse[col]:.4f}, RMSE: {avg_rmse[col]:.4f}")
    else:
        print("⚠️ No per-column metrics were computed. Check if masking and scaling are working correctly.")

    return avg_test_loss, avg_mae_value, avg_rmse_value, avg_mse_value, avg_mae, avg_rmse, avg_mse


def main():
    # Configuration parameters
    class Config:
        batch_size = 64
        epochs = 10
        no_cuda = False
        no_mps = False
        seed = 1
        log_interval = 10
        lr = 1e-4

    dl_kwargs = dict(
        batch_size=Config.batch_size,
        shuffle=True,
        # num_workers=4, # This may or may not be faster, depending on system
        pin_memory=torch.cuda.is_available(),
    )

    args = Config()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if args.cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    
    # Step 1: Apply masking
    train_masked, train_mask = mask_f(train_df)
    val_masked, val_mask = mask_f(val_df)
    test_masked, test_mask = mask_f(test_df)

    # Step 2: Scale the masked data, outputs are scaled and masked datasets
    train_scaled, val_scaled, test_scaled, data_min, data_max = preprocess(train_masked, val_masked, test_masked)


    #masked and scaled, should be passed into the train and test funcitons. 
    train_ms=torch.tensor(train_scaled.values,dtype=torch.float32)
    train_mask=torch.tensor(train_mask.values, dtype=torch.float32)

    val_ms = torch.tensor(val_scaled.values, dtype=torch.float32)
    val_mask = torch.tensor(val_mask.values, dtype=torch.float32)
    val_original=torch.tensor(val_df.values,dtype=torch.float32)

    test_ms= torch.tensor(test_scaled.values, dtype=torch.float32)
    test_mask = torch.tensor(test_mask.values, dtype=torch.float32)
    test_original=torch.tensor(test_df.values,dtype=torch.float32)



    #create datasets on CPU
    test_dataset=TensorDataset(test_ms,test_mask)  #Previous dataset 
    train_dataset=TensorDataset(train_ms,train_mask)
    val_dataset=TensorDataset(val_ms,val_mask)

     # Initialize Mlflow
    mlflow.set_experiment("VAE_final_test2")

  
    feature_names=data.columns.tolist()
    #Adding Optuna
    def objective(trial,device, data_min, scale):
        with mlflow.start_run(nested=True) as run:
            input_dim = train_scaled.shape[1]
            batch_size =trial.suggest_int("batch_size", 32, 128, step=32)
            lr = trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
        
            mlflow.log_params({
                    "batch_size":batch_size,
                    "learning_rate":lr,
                
                })

            model = VAE(input_dim=input_dim).to(device)

            optimizer = optim.Adam(model.parameters(), lr=lr)

            #best_val_loss=float('inf')
            for epoch in range(args.epochs):
                
                avg_train_loss = train(
                    epoch,
                    model,
                    train_loader,
                    optimizer=optimizer,
                    device=device,
                    args=args,
                
                )

                avg_test_loss, avg_mae_value, avg_rmse_value, avg_mse_value, avg_mae, avg_rmse, avg_mse = test(
                    epoch,
                    model,
                    data_min=data_min,
                    data_max=data_max,
                    test_loader=val_loader,
                    original=val_original,
                    device=device
                
                   )
                
            
                
            #early stopping
               # if avg_test_loss<best_val_loss:
                #    best_val_loss=avg_test_loss
              #  else:
              #      break
                
                mlflow.log_metrics(
                    {
                        "epoch": epoch,
                        "average_train_loss": avg_train_loss,
                        "average_test_loss": avg_test_loss,
                        "avg_mae_test": avg_mae_value,
                        "avg_rmse_test": avg_rmse_value,
                        "avg_mse_test":avg_mse_value,
                
                
                    },
                    step=epoch,
                 )

            trial.set_user_attr("model_state_dict", model.state_dict())
            trial.set_user_attr("mlflow_run_id", run.info.run_id)
            
        return avg_test_loss # Ensure only one value is returned

   
    # Define run name(edit)
    vae_run_name = "VAE_final_test_official"
    
    with mlflow.start_run(run_name=vae_run_name) as parent_run:
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial:objective (trial,device,data_min,data_max),n_trials=20, timeout=3600)
        best_trial = study.best_trial
        best_run_id = best_trial.user_attrs["mlflow_run_id"] #log run id to see the loss trend later
        best_batch_size=best_trial.params["batch_size"]
    
      
        mlflow.log_params(best_trial.params)
        mlflow.log_param("best_run_id", best_run_id)
        mlflow.log_metric("Best Validation Loss:", best_trial.value)
        print("Best Trial:")
        print(f"Validation loss: {best_trial.value}")
        print("Params:")


        for key, value in best_trial.params.items():
            print(f"{key}: {value}")


   #Newly revised part: After Optuna found the best params combo
    # Combine train and validation datasets
    combined_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
    combined_loader = DataLoader(combined_dataset, batch_size=best_batch_size, shuffle=True)

    #Retrain the model on combined dataset 
    # Instantiate a new model with best hyperparameters
    with mlflow.start_run(run_name="Final_model&val_for_all", nested=True):
        final_model = VAE(input_dim=train_scaled.shape[1]).to(device) #optimized
        optimizer = torch.optim.Adam(final_model.parameters(), lr=best_trial.params["learning_rate"])
           

        # Retrain on combined dataset
        epochs = 10  
        for epoch in range(epochs):
                
            train(
                epoch=epoch,
                model=final_model,
                train_loader=combined_loader,
                optimizer=optimizer,
                device=device,
                args=args,
            
                )

    #Test on the test set with best parameters
           
        #Evaluate on the test data 
        test_loader=DataLoader(test_dataset, batch_size=best_batch_size, shuffle=False)
        
        avg_test_loss, avg_mae_value, avg_rmse_value, avg_mse_value, avg_mae, avg_rmse, avg_mse = test(
                    test_loader=test_loader,
                    epoch=epochs,
                    model=final_model,
                    device=device,
                    original=test_original,
                    data_min=data_min,
                    data_max=data_max
                )
        # === Log overall average metrics ===
        mlflow.log_metrics({
            "epoch": epoch,
            "average_test_loss": avg_test_loss,
            "avg_mae_masked": avg_mae_value,
            "avg_rmse_masked": avg_rmse_value,
            "avg_mse_masked": avg_mse_value,
        }, step=epochs)

        percentage_mae_per_feature = {}

        for col_idx, feature in enumerate(feature_names):
            # Log masked metrics
            if col_idx in avg_mae:
                mlflow.log_metric(f"Masked_MAE_{feature}", avg_mae[col_idx])
            if col_idx in avg_mse:
                mlflow.log_metric(f"Masked_MSE_{feature}", avg_mse[col_idx])
            if col_idx in avg_rmse:
                mlflow.log_metric(f"Masked_RMSE_{feature}", avg_rmse[col_idx])

            masked_mae = avg_mae.get(col_idx)
            mask_i = test_mask[:, col_idx]
            ground_truth_i = test_original[:, col_idx]
            masked_values = ground_truth_i[mask_i == 1]

            if masked_mae is not None and masked_values.numel() > 1:
                std_dev_i = torch.std(masked_values, unbiased=True)
                if std_dev_i != 0:
                  #  percentage_mae_i = (masked_mae / std_dev_i) * 100
                    percentage_mae_i = ((masked_mae / std_dev_i) * 100).item()

                else:
                    percentage_mae_i = float('inf')
                percentage_mae_per_feature[col_idx] = percentage_mae_i
                mlflow.log_metric(f"Percentage_MAE_vs_STD_{feature}", percentage_mae_i)
            else:
                percentage_mae_per_feature[col_idx] = None  # or np.nan
                print(f"⚠️ Skipping Percentage_MAE_vs_STD for {feature} due to insufficient data.")


            mlflow.log_metric(f"Percentage_MAE_vs_STD_{feature}", percentage_mae_i)

          
        # === Print summary ===
        print(f"\n📊 Final Evaluation Results (Epoch {epoch})")
        print(f"Test Loss: {avg_test_loss:.4f}")
        print(f"Masked   - Avg MAE: {avg_mae_value:.4f}, Avg MSE: {avg_mse_value:.4f}, Avg RMSE: {avg_rmse_value:.4f}, Perc:{percentage_mae_i:.4f}")

        print("\n🔍 Per-Feature Metrics:")

        #load results in dataframe
        VAE_metrics_data = []
        for col_idx, feature in enumerate(feature_names):
            masked_mae = avg_mae.get(col_idx)
            masked_mse = avg_mse.get(col_idx)
            masked_rmse = avg_rmse.get(col_idx)
            perc_mae = percentage_mae_per_feature.get(col_idx)

         
            VAE_metrics_data.append({
                        "feature": feature,
                        "masked_mae": avg_mae.get(col_idx),
                        "masked_mse": avg_mse.get(col_idx),
                        "masked_rmse":avg_rmse.get(col_idx),
                        "percentage_mae_vs_std":percentage_mae_per_feature.get(col_idx)
                    })
     
            #save result as df
            metrics_df = pd.DataFrame(VAE_metrics_data)
            print(metrics_df)
            metrics_df.to_parquet("/home/ec2-user/SageMaker/sensor-imputation-thesis/src/sensor_imputation_thesis/nadire/VAE_metrics.parquet", index=False)

           
            print(f"\nFeature: {feature}")
            if masked_mae is not None:
                print(f"  [Masked]   MAE: {masked_mae:.4f}, MSE: {masked_mse:.4f}, RMSE: {masked_rmse:.4f}") 
            if perc_mae is not None:
                print(f" | %MAE vs STD={perc_mae:.2f}%")
            else:
                print()

           

        #Register Model 
        #log the model with scaling params
        input_dim = train_scaled.shape[1]
        input_example = torch.randn(1, input_dim).float().cpu().numpy()  # Convert to NumPy
            
        final_model.cpu()
        final_model.eval()

        mlflow.pytorch.log_model(
        pytorch_model=final_model,
        artifact_path="model",
        registered_model_name="VAE_All_Final_with_mcar",
        input_example=input_example,
        extra_pip_requirements=["torch","optuna","mlflow"],
        
                                                            
            metadata={
                "train_min": {k: float(v) for k, v in data_min.items()},
                "train_max": {k: float(v) for k, v in data_max.items()}
                }
            )
            


if __name__ == '__main__':
    main()


#After training and registering model, define a imputing function to directly impute the original data with missing values, 
#But to be cautious here, columns with all nans can not be included because model didn't learn to learn its pattern

#data to be imputed (with col having all nans dropped) and with numeric cols
df_with_nans=df.dropna(axis=1,how='all').select_dtypes(include='number')
#check df len, it has same rows with the original dataframe
len(df_with_nans)

#Define imputation function (directly fill in the missing values in df_toimpute )


def impute_with_vae(df_with_nans, model, feature_names, device, feature_range=(-1, 1)):
    """
    Imputes missing values in a DataFrame using a trained VAE model.
    Automatically scales and inverse-scales the data per feature.

    Parameters:
    - model: Trained VAE model.
    - df_with_nans: DataFrame with missing values (NaNs).
    - device: Torch device.
    - feature_names: List of feature names in the same order as model input.
    - feature_range: Tuple for MinMax scaling (default: (-1, 1)).

    Returns:
    - imputed_df_original_scale: DataFrame with imputed values in original scale.
    """

    df = df_with_nans.copy()
    df_scaled = df_with_nans.copy()
    scalers = {}

    # === Scale each feature individually ===
    for col in feature_names:
        scaler = MinMaxScaler(feature_range=feature_range)
        col_values = df[col].values.astype(np.float32)
        non_nan_mask = ~np.isnan(col_values)

        if non_nan_mask.sum() == 0:
            continue  # Skip columns with all NaNs

        # Fit scaler on non-NaN values
        scaler.fit(col_values[non_nan_mask].reshape(-1, 1))

        # Transform and assign scaled values
        scaled_col = col_values.copy()
        scaled_col[non_nan_mask] = scaler.transform(col_values[non_nan_mask].reshape(-1, 1)).flatten()
        df_scaled[col] = scaled_col
        scalers[col] = scaler

    # === Impute missing values using VAE ===
    with torch.no_grad():
        for i, row in df_scaled.iterrows():
            if row.isna().any():
                input_data = row.fillna(0).values.astype(np.float32)
                input_tensor = torch.tensor(input_data).unsqueeze(0).to(device)

                mu, logvar = model.encode(input_tensor)
                z = model.reparameterize(mu, logvar)
                recon_row = model.decode(z).cpu().numpy().flatten()

                for col in row.index[row.isna()]:
                    df_scaled.at[i, col] = recon_row[feature_names.index(col)]

    # === Inverse scale to original range ===
    df_imputed = df_scaled.copy()
    for col in feature_names:
        if col in scalers:
            col_values = df_scaled[col].values.reshape(-1, 1).astype(np.float32)
            df_imputed[col] = scalers[col].inverse_transform(col_values).flatten()

    return df_imputed


#Apply the function
#log the model
model_name = "VAE_All_Final_with_mcar"
version = "1" 
client = mlflow.tracking.MlflowClient()

# Promote the model version to Production
client.transition_model_version_stage(
    name=model_name,
    version=version,
    stage="Production"
)

# Load the model from the Production stage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_uri = f"models:/{model_name}/Production"
vae_best_model = mlflow.pytorch.load_model(model_uri).to(device)
vae_best_model.eval()

imputed_df = impute_with_vae(
    model=vae_best_model,
    df_with_nans=df_with_nans,
    device=device,
    feature_names=feature_names
)

# View results
print("Imputed (scaled):")
print(imputed_df.head())
file_path="/home/ec2-user/SageMaker/sensor-imputation-thesis/src/sensor_imputation_thesis/nadire/DFImputedwizVAE"
imputed_df.to_parquet(f'{file_path}.parquet')
print("\nImputed (original scale):")
print(df_with_nans.head())
df_with_nans.to_parquet("/home/ec2-user/SageMaker/sensor-imputation-thesis/src/sensor_imputation_thesis/nadire/df_wiz_NaNs.parquet")



