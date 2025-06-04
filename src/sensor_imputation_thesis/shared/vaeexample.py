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
from data_insight import setup_duckdb
from duckdb import DuckDBPyConnection as DuckDB
from duckdb import DuckDBPyRelation as Relation
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from tqdm.auto import tqdm

import sensor_imputation_thesis.shared.load_data as load

# set tracking_uri:
mlflow.set_tracking_uri("http://localhost:5000")
# Initialize Mlflow
mlflow.set_experiment("VAE Training_test")

# So I don't know above commands are sufficient server setup, but it did work for me. And I might need your help if they aren't
# mlflow relevant info:
# I overall faced problem that says "HTTPConnectionPool(host='0.0.0.0', port=5000): Max retries exceeded with url". And the solutions were:
# 1. Try running this line in terminal to see if mlflow logs exist
# if logs exist, port can be directed
# " mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns "
# 2. Another possible solution for mlflow errors is to rename the experiment name and the run-name involved below

# Load data
# Here for training vae, tag list for 9 missing cols is used.

pd.set_option("display.max_columns", None)


def load_engine_data(
    con: DuckDB, product_id: str, start: pd.Timestamp, stop: pd.Timestamp, tags: list[str]
) -> Relation:
    return con.sql(f"""
    SELECT {",".join(tags)}
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

# All preprocessing steps(drop columns with different indexes)
df1 = df.drop(
    columns=[
        "te_exh_cyl_out__1",
        "te_exh_cyl_out__2",
        "te_exh_cyl_out__3",
        "te_exh_cyl_out__4",
        "te_exh_cyl_out__5",
        "te_exh_cyl_out__6",
    ]
)
# filter df with engine running
df1 = df1[df1["fr_eng"] > 0] # suggest higher number here (eg, 10/60) revolutions per seconds 

# Check nan values in each column
for col in df1.columns:
    print(f"Column {col} has {df1[col].isna().sum()} NaN values")

# Updated Function: Assign dummy values to nan columns/values


def impute_nan(data, target_col):
    """
    Replace NaN values with dummy values based on specified criteria.

    Args:
       data: Input DataFrame
       target_col: The target column name that needs a single dummy value for the entire column.

    Returns:
      Data with NaNs replaced by dummy values
    """
    df = data.copy()

    for col in df.columns:
        # Assign one single dummy to the whole target_col
        if target_col in df.columns:
            df[target_col] = -2

        # if no nan, skip
        if df[col].isna().sum() == 0:
            continue

        # same dummy for the col with nan values (all & partial)
        if df[col].isna().sum() > 0:
            # Columns with partial NaN values, assign dummy value -2
            df.loc[df[col].isna(), col] = -2

    return df


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

# Impute NaN values
data = impute_nan(scaled_df, "te_exh_cyl_out__0")


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

# Split the dataset into training and testing sets based on time
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])


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
targetcolumn = "te_exh_cyl_out__0"
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


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in (pbar := tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0)):
        data = data.to(device).float()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        lambda_val = 0.7  # (for finding optimized lambda, grid search might be applied, but now I just set it as 0.7)
        loss = loss_function(recon_batch, data, mu, logvar, targetcol_idx, lambda_val)
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
    print("====> Epoch: {} Average train loss: {:.4f}".format(epoch, average_train_loss))
    return average_train_loss


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device).float()  # Convert to FloatTensor
            recon_batch, mu, logvar = model(data)
            # define new lambda val for testing
            lambda_val = 1.0
            test_loss += loss_function(recon_batch, data, mu, logvar, targetcol_idx, lambda_val).item()
    test_loss /= len(test_loader.dataset)
    print("====> Test set loss: {:.4f}".format(test_loss))
    average_test_loss = test_loss / len(test_loader.dataset)
    print("====> Epoch: {} Average test loss: {:.4f}".format(epoch, average_test_loss))
    return average_test_loss


# Define runn ame
vae_run_name = "vaetest_for_te_exh_cyl_out__0"

if __name__ == "__main__":
    with mlflow.start_run(run_name=vae_run_name):
        for epoch in range(1, args.epochs + 1):
            average_train_loss = train(epoch)
            average_test_loss = test(epoch)

            mlflow.log_metrics(
                {
                    "epoch": epoch,
                    "average_train_loss": average_train_loss,
                    "average_test_loss": average_test_loss,
                },
                step=epoch,
            )

            mlflow.log_params({"latent_dim": scaled_data.shape[1], "lr": Config.lr, "batch_size": Config.batch_size})

            mlflow.end_run()

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

    # Density plot

    # Filter the original and imputed data for the specified column. If one wants to plot on whole df, change code here.
    orig_col_filtered = numeric_df[targetcolumn].dropna()
    imputed_col_filtered = generated_df[targetcolumn].dropna()

    # Create the density plot
    plt.figure(figsize=(10, 6))
    sns.kdeplot(orig_col_filtered, fill=True, label="Original")
    sns.kdeplot(imputed_col_filtered, fill=True, label="Imputed")
    plt.title(f"Density Plot of {targetcolumn}")
    plt.legend()
    plt.tight_layout()
    plt.show()
