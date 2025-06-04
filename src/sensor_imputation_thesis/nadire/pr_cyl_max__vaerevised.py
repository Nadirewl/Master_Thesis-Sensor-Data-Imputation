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
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

import sensor_imputation_thesis.shared.load_data as load


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# set tracking_uri:
mlflow.set_tracking_uri("http://localhost:5000")
# Initialize Mlflow
mlflow.set_experiment("VAE_adjustedapproach")

# So I don't know above commands are sufficient server setup, but it did work for me. And I might need your help if they aren't
# mlflow relevant info:
# I overall faced problem that says "HTTPConnectionPool(host='0.0.0.0', port=5000): Max retries exceeded with url". And the solutions were:
# 1. Try running this line in terminal to see if mlflow logs exist
# if logs exist, port can be directed
# " mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns "
# 2. Another possible solution for mlflow errors is to rename the experiment name and the run-name involved below

# Load data

pd.set_option("display.max_columns", None)


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
    return hashlib.md5(",".join(tags).encode()).hexdigest()


# edit timestamp as needed
start, stop = pd.Timestamp("2023-10-01"), pd.Timestamp("2024-10-01")

# these are the tags that include 9 missing cols, feel free to edit
tags = [
    "time",
    "fr_eng",
    "te_exh_cyl_out__0",
    "te_exh_cyl_out__1",
    "te_exh_cyl_out__2",
    "te_exh_cyl_out__3",
    "te_exh_cyl_out__4",
    "te_exh_cyl_out__5",
    "te_exh_cyl_out__6",
    "pd_air_ic__0",
    "pr_exh_turb_out__0",
    "pr_exh_turb_out__1",
    "pr_exh_turb_out__2",
    "pr_exh_turb_out__3",
    "te_air_ic_out__0",
    "te_air_ic_out__1",
    "te_air_ic_out__2",
    "te_air_ic_out__3",
    "te_seawater",
    "te_air_comp_in_a__0",
    "te_air_comp_in_a__1",
    "te_air_comp_in_a__2",
    "te_air_comp_in_a__3",
    "fr_tc__0",
    "fr_tc__1",
    "fr_tc__2",
    "fr_tc__3",
    "pr_baro",
    "te_exh_turb_in__0",
    "te_exh_turb_in__1",
    "te_exh_turb_in__2",
    "te_exh_turb_in__3",
    "te_exh_turb_out__0",
    "te_exh_turb_out__1",
    "te_exh_turb_out__2",
    "te_exh_turb_out__3",
    "pr_exh_rec",
    "pr_air_scav",
    "pr_air_scav_ecs",
    "fr_eng_setpoint",
    "te_air_ic_out__0",
    "te_air_ic_out__1",
    "te_air_ic_out__2",
    "te_air_ic_out__3",
    "pd_air_ic__0",
    "pd_air_ic__1",
    "pd_air_ic__2",
    "pd_air_ic__3",
    "pr_cyl_max__0",
    "pr_cyl_max__1",
    "pr_cyl_max__2",
    "pr_cyl_max__3",
    "pr_cyl_max__4",
    "pr_cyl_max__5",
    "pr_cyl_max__6",
    "pr_cyl_max__7",
    "pr_cyl_max__8",
    "pr_cyl_max__9",
    "pr_cyl_max__10",
    "pr_cyl_max__11",
    #more tags
    "se_mip_acco__0",
    "se_mip_acco__1",
    "se_mip_acco__2",
    "se_mip_acco__3",
    "se_mip_acco__4",
    "se_mip_acco__5",
    "fr_eng_ecs",
    #"re_eng_load_ecs", #not found
    # "re_eng_load_pmi", #not found
    "se_mip__0",
    "se_mip__1",
    "se_mip__2",
    "se_mip__3",
    "se_mip__4",
    "se_mip__5",
    "se_mip__6",
    "se_mip__7",
    "se_mip__8",
    "se_mip__9",
    "se_mip__10",
    "se_mip__11",
    "pr_cyl_comp__0",
    "pr_cyl_comp__1",
    "pr_cyl_comp__2",
    "pr_cyl_comp__3",
    "pr_cyl_comp__4",
    "pr_cyl_comp__5",
    "pr_cyl_comp__6",
    "pr_cyl_comp__7",
    "pr_cyl_comp__8",
    "pr_cyl_comp__9",
    "pr_cyl_comp__10",
    "pr_cyl_comp__11",
    "te_cw_ic_in_common",
    "te_air_ic_out__0",
    "te_air_ic_out__1",
    "te_air_ic_out__2",
    "te_air_ic_out__3",
    "in_stable", 
    "te_exh_turb_in_iso__0", 
    "te_exh_turb_in_iso__1",
    "te_exh_turb_in_iso__2",
    "te_exh_turb_in_iso__3",
    "fr_tc_iso__0", 
    "fr_tc_iso__1",
    "fr_tc_iso__2",
    "fr_tc_iso__3",
    "pr_cyl_max_mv_iso", 
    "pr_cyl_comp_mv_iso", 
    "pr_air_scav_iso", 
    # tags needed for sfoc score calculation
    "te_air_scav_rec_iso", 
    "re_perf_idx_hrn_indicator", 
    "in_engine_running_mode",
    "te_exh_turb_in__0",
    "te_exh_turb_in__1",
    "te_exh_turb_in__2",
    "te_exh_turb_in__3",
    "bo_aux_blower_running",
    "re_eng_load",

]

# edit point as needed
product_id = "89ccb7a888d53f8792f0580801cede9a"


# Construct the cache file path using the Path class
DATA = Path.home() / "SageMaker/data_cache"
DATA.mkdir(exist_ok=True)
cache = DATA / f"data_{get_tags_hash(tags)}_{product_id}_{start:%Y-%m-%d}_{stop:%Y-%m-%d}.parquet"

# Ensure the directory exists
cache.parent.mkdir(parents=True, exist_ok=True)

if cache.exists():
    df = pd.read_parquet(cache)
else:
    con = setup_duckdb()
    df = load_engine_data(con, product_id, start, stop, tags).df()
    df.to_parquet(cache)

print(df.head(10))


## Adding engine types and assign onehot encoder, and merge df  
con = data_insight.setup_duckdb()
con.sql("SET enable_progress_bar=true")
engine_type = con.sql(f"SELECT engine_type FROM shipinfo WHERE productId == '{product_id}'").df().engine_type.item()
df['engine_type']=engine_type

#Apply onehotencoder 
encoder = OneHotEncoder(sparse_output=False)
encoded_engine_type = encoder.fit_transform(df[['engine_type']])
# Create a DataFrame with the encoded columns
encoded_df = pd.DataFrame(encoded_engine_type, columns=encoder.get_feature_names_out(['engine_type']))
# Concatenate the original DataFrame with the encoded DataFrame
df = pd.concat([df, encoded_df], axis=1)
#drop? not sure yet. 
df.drop('engine_type', axis=1,inplace=True)



# All preprocessing steps(drop columns with different indexes if there are several indexes); Here the target column is changed 
#into pr_cyl_max
df1 = df.drop(
    columns=[
        "pr_cyl_max__1",
        "pr_cyl_max__2",
        "pr_cyl_max__3",
        "pr_cyl_max__4",
        "pr_cyl_max__5",
        "pr_cyl_max__6",
        "pr_cyl_max__7",
        "pr_cyl_max__8",
        "pr_cyl_max__9",
        "pr_cyl_max__10",
        "pr_cyl_max__11",
    ]
)
# filter df with engine running (changed into 10/60 revolutions)
df1 = df1[df1["fr_eng"] > (10/60)]

# Check nan values in each column
for col in df1.columns:
    print(f"Column {col} has {df1[col].isna().sum()} NaN values")


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

if numeric_df.empty:
    print("The DataFrame is empty. Please provide valid data.")
else:
    # Standardize the numeric data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(numeric_df)
    # Convert scaled data back to DataFrame
    scaled_df = pd.DataFrame(scaled_data, columns=numeric_df.columns)

data=scaled_df


# Custom dataset for float data
class FloatDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Ensure integer-based indexing
        return torch.tensor(self.data[idx], dtype=torch.float32)


# Create DataLoader
dataset = FloatDataset(data.values)
dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

# Split the dataset into training, validation, and testing sets based on time
#tscv=TimeSeriesSplit(n_splits=5,test_size=int(0.2 * len(dataset)))

train_size = int(0.6 * len(dataset))
val_size=int(0.2 * len(dataset))
test_size = len(dataset) - (train_size + val_size)

train_dataset = dataset[:train_size]
val_dataset = dataset[train_size:train_size + val_size]
test_dataset = dataset[train_size + val_size:]


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
train_loader = DataLoader(train_dataset, **dl_kwargs)
test_loader = DataLoader(test_dataset, **dl_kwargs)
val_loader=DataLoader(val_dataset,**dl_kwargs)


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


# VAE Model
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
        return self.fc4(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE(input_dim=scaled_data.shape[1]).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Specify the target column, here should be changed while training other target columns
# (edit point as needed)
targetcolumn = "pr_cyl_max__0"
targetcol_idx = data.columns.get_loc(targetcolumn)


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


###add mse(&mae) comparison within training loop for target column
def train(epoch,lambda_val):
    model.train()
    train_loss = 0
    mse_loss_target_train=0
    mae_loss_target_train=0
    for batch_idx, data in (pbar := tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0)):
        data_unmasked = data.to(device).float()
        #mask the target column
        data_masked=data_unmasked.clone()
        data_masked[:,targetcol_idx]=-2
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data_masked) #masked target col
        loss = loss_function(recon_batch, data_unmasked, mu, logvar, targetcol_idx, lambda_val)
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"NaN or Inf detected in loss at batch {batch_idx}.")
            continue
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
        average_train_loss = train_loss / len(train_loader.dataset) 
        target_column = data_unmasked[:, targetcol_idx].cpu().numpy()
        recon_column = recon_batch[:, targetcol_idx].cpu().detach().numpy()
        mse_loss_target_train += mean_squared_error(target_column, recon_column) #ground truth
        mae_loss_target_train += mean_absolute_error(target_column,recon_column)
    print("===> Epoch: {} Average mse test loss for targetcol :{:.4f}".format(epoch,mse_loss_target_train/len(train_loader)))
    print("===> Epoch: {} Average mae test loss for targetcol :{:.4f}".format(epoch,mae_loss_target_train/len(train_loader)))
    print("====> Epoch: {} Average train loss: {:.4f}".format(epoch, average_train_loss))
    return average_train_loss, mse_loss_target_train / len(train_loader), mae_loss_target_train / len(train_loader)


#Added mse in the test
# In the test loop, masked column should be the input. 

#Redefine data for the test loop(should I inlcude this in the test loop or leave it here?)

#dataset changed into validation set for testing loop
def test(epoch):
    model.eval()
    test_loss = 0
    mse_loss_target_test=0
    mae_loss_target_test=0
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            data_unmasked = data.to(device).float()
            #mask the target column
            data_masked=data_unmasked.clone()
            data_masked[:,targetcol_idx]=-2
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data_masked) #masked target col
            # define new lambda val for testing
            lambda_val = 1.0
            test_loss += loss_function(recon_batch, data_unmasked, mu, logvar, targetcol_idx, lambda_val).item()
            #If we only want to focus on the target column
            target_column = data_unmasked[:, targetcol_idx].cpu().numpy()
            recon_column = recon_batch[:, targetcol_idx].cpu().detach().numpy()
            mse_loss_target_test += mean_squared_error(target_column, recon_column)
            mae_loss_target_test += mean_absolute_error(target_column,recon_column)
    print("===> Epoch: {} Average mse test loss for targetcol: {:.4f}".format(epoch, mse_loss_target_test / len(val_loader)))
    print("===> Epoch: {} Average mae test loss for targetcol: {:.4f}".format(epoch, mae_loss_target_test / len(val_loader)))
    print("====> Test set loss: {:.4f}".format(test_loss))
    average_test_loss = test_loss / len(val_loader.dataset)
    print("====> Epoch: {} Average test loss: {:.4f}".format(epoch, average_test_loss)) 
    return average_test_loss, mse_loss_target_test / len(val_loader), mae_loss_target_test / len(val_loader)
    


# Define run name
vae_run_name = "vae(adjusted)_for_pr_cyl_max__0"


#Adding Optuna
def objective(trial):
     input_dim = scaled_data.shape[1]
     batch_size = trial.suggest_int("batch_size", 32, 128, step=32)
     learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
     lambda_val = trial.suggest_float("lambda", 0.5, 1.0)

     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

     model = VAE(input_dim=input_dim).to(device)
     optimizer = optim.Adam(model.parameters(), lr=learning_rate)

     for epoch in range(args.epochs):
         train_loss, mse_loss_target_train, mae_loss_target_train = train(epoch, lambda_val)
         val_loss, mse_loss_target_test, mae_loss_target_test = test(epoch)

     return val_loss  # Ensure only one value is returned

if __name__ == "__main__":
     study = optuna.create_study(direction="minimize")
     study.optimize(objective, n_trials=100, timeout=3600)
     #plot_optimization_history(study)  #visualization 
     best_trial = study.best_trial

     with mlflow.start_run(run_name=vae_run_name):
         mlflow.log_params(best_trial.params)
         mlflow.log_metric("validation_loss", best_trial.value)  # Log the single value

         for epoch in range(1, args.epochs + 1):
             average_train_loss, mse_loss_target_train, mae_loss_target_train = train(epoch, best_trial.params['lambda'])
             average_test_loss, mse_loss_target_test, mae_loss_target_test = test(epoch)

             mlflow.log_metrics(
                 {
                     "epoch": epoch,
                     "average_train_loss": average_train_loss,
                     "average_test_loss": average_test_loss,
                     "mse_loss_target_train": mse_loss_target_train,
                     "mae_loss_target_train": mae_loss_target_train,
                     "mse_loss_target_test": mse_loss_target_test,
                     "mae_loss_target_test": mae_loss_target_test
                 },
                 step=epoch,
             )

         mlflow.end_run()

     print("Best Trial:")
     print(f"Validation loss: {best_trial.value}")
     print("Params:")
     for key, value in best_trial.params.items():
         print(f"{key}: {value}")
 


    # CSOM: This should be done ONCE after training, and not inside the training loop.

    # synthesize data, not sure if this should be in the different script.
    # Feel free to remove the following part (including plt) if not needed.
with torch.no_grad():
        # Generate a latent sample
        sample = torch.randn(Config.batch_size, 20).to(device)  # Adjust latent dimensions
        # Decode the sample
        sample = model.decode(sample).cpu().numpy()
        # Validate the shape of the decoded sample
        if sample.shape[1] != len(numeric_df.columns):
            raise ValueError(
                f"Shape mismatch: Generated data has {sample.shape[1]} columns, but expected {len(numeric_df.columns)}."
            )

        # Reverse the scaling to map back to the original distribution
        generated_df = pd.DataFrame(scaler.inverse_transform(sample), columns=numeric_df.columns)
        print(generated_df.head())

print(generated_df.isnull().values.any())

    # Density plot

    # Filter the original and imputed data for the specified column. If one wants to plot on whole df, change code here.
#orig_col_filtered = numeric_df[targetcolumn].dropna()
#imputed_col_filtered = generated_df[targetcolumn].dropna()

    # Create the density plot
    #plt.figure(figsize=(10, 6))
    #sns.kdeplot(orig_col_filtered, fill=True, label="Original")
    #sns.kdeplot(imputed_col_filtered, fill=True, label="Imputed")
    #plt.title(f"Density Plot of {targetcolumn}")
    #plt.legend()
    #plt.tight_layout()
    #plt.show()


