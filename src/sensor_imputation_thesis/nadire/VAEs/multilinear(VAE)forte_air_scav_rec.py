import argparse
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import shap
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
df=pd.read_parquet("/home/ec2-user/SageMaker/sensor-imputation-thesis/src/sensor_imputation_thesis/nadire/df_for_simplemodels.parquet")
# Check nan values in each column
for col in df.columns:
    print(f"Column {col} has {df[col].isna().sum()} NaN values")


## Newly Suggested Approach to drop nan columns (drop cols with all nan values, and then drop rows that have any nan value )
df1=df.dropna(axis=1,how='all')
# drop any null vals in rows 
df1=df1.dropna(axis=0,how='any')


#check the length of the df1 and the length of df to check how many cols rows are dropped 
print("shape of original df:")
print("Datasize:",len(df))
print("Colnumber:",len(df.columns))

print("shape of filtered df:")
print("Datasize:",len(df1))
print("Colnumber:",len(df1.columns))

# Select numeric columns
numeric_df = df1.select_dtypes(include="number")
print(numeric_df)

data=numeric_df  #data here is before scaling 


#define inverse scale
def inverse_scale(scaled_data, data_min, scale):
    return scaled_data*scale + data_min


# Simply Multilinear
class VAE(nn.Module):
    def __init__(self, input_dim):  #input_dim might be changed into how many nodes in the layers
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, 400)
        self.fc2 = nn.Linear(400, 200)
        self.fc21 = nn.Linear(200, 40)
        self.fc22 = nn.Linear(200, 40)
        self.fc3 = nn.Linear(40, 400)
        self.fc4 = nn.Linear(400, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1)) #added one more layer (h2) 
        return self.fc21(h2), self.fc22(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.tanh(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
       # z = self.reparameterize(mu, logvar) 
        z=mu
        return self.decode(z), mu, logvar


# Reconstruction + KL divergence losses summed over target column and batch
def loss_function(recon_x, x, mu, logvar, targetcol_idx, lambda_val):
    # Specify the target column
    recon_x_col = recon_x[:, targetcol_idx]
    x_col = x[:, targetcol_idx]

    # MSE & KLD for target column
    MSE_col = F.mse_loss(recon_x_col, x_col, reduction="sum")
    KLD_col = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # MSE & KLD for entire dataset
    MSE_all = F.mse_loss(recon_x, x, reduction="sum")
    KLD_all = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Combine losses using lambda
    loss = lambda_val * (MSE_col + KLD_col) + (1 - lambda_val) * (MSE_all + KLD_all)
    return loss



def train(epoch, model:VAE, train_loader, data_min, scale, lambda_val, targetcol_idx, optimizer, device, args):
    model.train()
    train_loss = 0
    mae_loss_target_train = 0
    average_train_loss = 0  # Initialize average_train_loss
    pbar = tqdm(total=len(train_loader), desc="Training")
    for batch_idx, (data,) in enumerate(train_loader): #trian loader is after scaling
        data_unmasked = data.to(device).float() #this data is before scaling
        # Mask the target column
        data_masked = data_unmasked.clone() #before scaling
        data_masked[:, targetcol_idx] = -2
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data_masked)  # Masked target col
        if torch.isnan(recon_batch).any():
            print("NaN or in recon_batch!")
            break
        loss = loss_function(recon_batch, data_unmasked, mu, logvar, targetcol_idx, lambda_val)
        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError(f"NaN or Inf detected in loss at batch {batch_idx}. Check scaling or model outputs!")
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
        #revised 
        data_unscaled=inverse_scale(data_unmasked,data_min.to(device),scale.to(device))
        target_column=data_unscaled[:,targetcol_idx]
        mu_nonoise, _ =model.encode(data_masked)
        recon_batch_nonoise =model.decode(mu_nonoise)
        recon_unscaled=inverse_scale(recon_batch_nonoise,data_min.to(device),scale.to(device))
        recon_column=recon_unscaled[:,targetcol_idx]

                                                                                                
        #mae metric calculation
        mae_loss_target_train+=torch.mean(torch.abs(target_column-recon_column)).item()
        #mae_loss_target_train += mean_absolute_error(target_column, recon_column)  # Ground truth

    average_train_loss = train_loss / len(train_loader.dataset)  # Update after loop
    print("====> Epoch: {} Average train loss: {:.4f}".format(epoch, average_train_loss))
    print("===> Epoch: {} Average mae train loss for targetcol :{:.4f}".format(epoch,mae_loss_target_train/len(train_loader)))
    return average_train_loss, mae_loss_target_train / len(train_loader)


#Added mse in the test
#In the test loop, masked column should be the input. 
#Redefine data for the test loop(should I inlcude this in the test loop or leave it here?)
#dataset changed into validation set for testing loop

def test(epoch, model:VAE, val_loader, scale, data_min, targetcol_idx, device):
    model.eval()
    test_loss = 0
    mae_loss_target_test = 0
    mse_loss_target_test = 0
    rmse_loss_target_test = 0

    with torch.no_grad():
        for batch_idx, (data,) in enumerate(val_loader):
            data_unmasked = data.to(device).float()  #data used here is before scaling 
            # Mask the target column
            data_masked = data_unmasked.clone()
            data_masked[:, targetcol_idx] = -2
            recon_batch, mu, logvar = model(data_masked)  # Masked target col
            # Define new lambda val for testing
            lambda_val = 1.0
            test_loss += loss_function(recon_batch, data_unmasked, mu, logvar, targetcol_idx, lambda_val).item()
            # Scale back target column and reconstructed column
            data_unscaled=inverse_scale(data_unmasked,data_min.to(device),scale.to(device))
            target_column=data_unscaled[:,targetcol_idx]
            # Masked target col
            mu_nonoise, _ =model.encode(data_masked)
            recon_batch_nonoise =model.decode(mu_nonoise)
            recon_unscaled=inverse_scale(recon_batch_nonoise,data_min.to(device),scale.to(device))

            recon_column=recon_unscaled[:,targetcol_idx]
                                                           
            #mae metric calculation
            mae_loss_target_test+=torch.mean(torch.abs(target_column-recon_column)).item()
            mse_loss_target_test += torch.mean((target_column - recon_column) ** 2).item()
            rmse_loss_target_test += torch.sqrt(torch.mean((target_column - recon_column) ** 2)).item()
        
    print("===> Epoch: {} Average mae test loss for targetcol: {:.4f}".format(epoch, mae_loss_target_test / len(val_loader)))
    print("====> Test set loss: {:.4f}".format(test_loss))
    average_test_loss = test_loss / len(val_loader.dataset)
    print("====> Epoch: {} Average test loss: {:.4f}".format(epoch, average_test_loss))
    return average_test_loss, mae_loss_target_test / len(val_loader), rmse_loss_target_test/len(val_loader), mse_loss_target_test/len(val_loader)



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
    
   # Specify the target column, here should be changed while training other target columns
    # (edit point as needed)
    targetcolumn = "te_air_scav_rec"
    targetcol_idx = data.columns.get_loc(targetcolumn)

    #split dataset
    train_size = int(0.6 * len(data))
    val_size=int(0.2 * len(data))

    train_df = data.iloc[:train_size]
    val_df = data.iloc[train_size:train_size + val_size]
    test_df= data.iloc[train_size + val_size:]

    # Standardize the numeric data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    #only apply scaler to the train set 
    scaled_train=scaler.fit_transform(train_df.values)
    print("scaled_data", scaled_train)
    print("unscaled_data",data)

    # Transform val and test using the same scaler
    scaled_val=scaler.transform(val_df.values)
    scaled_test=scaler.transform(test_df.values)

        
    data_min=torch.tensor(scaler.data_min_,dtype=torch.float32)
    data_max=torch.tensor(scaler.data_max_,dtype=torch.float32)
    scale=data_max-data_min
    scale[scale==0]=1e-8


    #Convert DataFrame to Tensor
    scaled_train_tensor=torch.tensor(scaled_train,dtype=torch.float32)
    scaled_val_tensor=torch.tensor(scaled_val,dtype=torch.float32)
    scaled_test_tensor=torch.tensor(scaled_test,dtype=torch.float32)

    #create datasets on CPU
    train_dataset=TensorDataset(scaled_train_tensor)
    val_dataset=TensorDataset(scaled_val_tensor)
    test_dataset=TensorDataset(scaled_test_tensor)


     # Initialize Mlflow
    mlflow.set_experiment("MultiLinear_for_te_air_scav_rec")



    #Adding Optuna
    def objective(trial,device, data_min, scale):
        with mlflow.start_run(nested=True) as run:
            input_dim = scaled_train.shape[1]
            batch_size =trial.suggest_int("batch_size", 32, 128, step=32)
            lr = trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True)
            lambda_val = trial.suggest_float("lambda", 0.5, 1.0)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
            
            mlflow.log_params({
                    "batch_size":batch_size,
                    "learning_rate":lr,
                    "lambda":lambda_val
                })

            model = VAE(input_dim=input_dim).to(device)

            optimizer = optim.Adam(model.parameters(), lr=lr)


            for epoch in range(args.epochs):
                
                average_train_loss, mae_loss_target_train = train(
                    epoch,
                    model,
                    train_loader,
                    lambda_val=lambda_val,
                    targetcol_idx=targetcol_idx,
                    optimizer=optimizer,
                    device=device,
                    data_min=data_min, 
                    scale=scale, #this is used for inverse scaling
                    args=args
                )

                average_test_loss, mae_loss_target_test,rmse_loss_target_test,mse_loss_target_test=test(
                    epoch,
                    model,
                    val_loader,
                    targetcol_idx=targetcol_idx,
                    device=device,
                    data_min=data_min,
                    scale=scale #same, used for inverse scaling 
                
                )
                

                
                mlflow.log_metrics(
                    {
                        "epoch": epoch,
                        "average_train_loss": average_train_loss,
                        "average_test_loss": average_test_loss,
                        "mae_loss_target_train": mae_loss_target_train,
                        "mae_loss_target_test": mae_loss_target_test,
                        "mse_loss_target_test":mse_loss_target_test,
                        "rmse_loss_target_test":rmse_loss_target_test
                    },
                    step=epoch,
)


            trial.set_user_attr("model_state_dict", model.state_dict())
            trial.set_user_attr("mlflow_run_id", run.info.run_id)
            
            return average_test_loss # Ensure only one value is returned

   
    # Define run name(edit)
    vae_run_name = "Multilinear_for_te_air_scav_rec"
    
    with mlflow.start_run(run_name=vae_run_name) as parent_run:
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial:objective (trial,device,data_min,scale),n_trials=20, timeout=3600)
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
    with mlflow.start_run(run_name="Final_model&val_for_pr_baro", nested=True):
        final_model = VAE(input_dim=scaled_train.shape[1]).to(device) #optimized
        optimizer = torch.optim.Adam(final_model.parameters(), lr=best_trial.params["learning_rate"])
           

        # Retrain on combined dataset
        epochs = 10  
        for epoch in range(epochs):
                
            train(
                epoch=epoch,
                model=final_model,
                train_loader=combined_loader,
                lambda_val=best_trial.params["lambda"],
                targetcol_idx=targetcol_idx,
                optimizer=optimizer,
                device=device,
                data_min=data_min,
                scale=scale, #this is used for inverse scaling
                args=args
                )

    #Test on the test set with best parameters
              
        #Evaluate on the test data 
        test_loader=DataLoader(test_dataset, batch_size=best_batch_size, shuffle=False)
        average_test_loss, mae_loss_target_test,rmse_loss_target_test,mse_loss_target_test=test(
                    val_loader=test_loader,
                    epoch=epochs,
                    model=final_model,
                    targetcol_idx=targetcol_idx,
                    device=device,
                    data_min=data_min,
                    scale=scale
                    )
                
                
        mlflow.log_metrics(
                {
                "epoch": epochs,
                "Final_test_loss": average_test_loss,
                "Final_mae_target_test": mae_loss_target_test,
                "Final_rmse_target_test":rmse_loss_target_test,
                "Final_mse_target_test":mse_loss_target_test
                    },
                step=epochs,
                )
           
        print(f"Final Test Loss:{average_test_loss:.4f}, MAE:{mae_loss_target_test:.4f},MSE:{mse_loss_target_test:.4f}, RMSE:{rmse_loss_target_test}")
            

            #Register Model 
            #log the model with scaling params
        input_dim = scaled_train.shape[1]
        input_example = torch.randn(1, input_dim).float().cpu().numpy()  # Convert to NumPy
            
        final_model.cpu()
        final_model.eval()

        mlflow.pytorch.log_model(
        pytorch_model=final_model,
        artifact_path="model",
        registered_model_name="Multilinear_derivedfraVAE_teairscavrec",
        input_example=input_example,
        extra_pip_requirements=["torch","optuna","mlflow"],
                                                             
            metadata={
                "train_min": data_min.cpu().numpy().tolist(),
                "scale": scale.cpu().numpy().tolist()
                }
            )
            

if __name__ == '__main__':
    main()







    # CSOM: This should be done ONCE after training, and not inside the training loop.

    