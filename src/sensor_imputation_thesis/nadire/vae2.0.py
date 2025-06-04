import sensor_imputation_thesis.shared.load_data as load 
from data_insight import setup_duckdb
from duckdb import DuckDBPyConnection as DuckDB
import pandas as pd
from duckdb import DuckDBPyRelation as Relation
from pathlib import Path
import hashlib
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

#Read data presaved
df=pd.read_parquet("/home/ec2-user/SageMaker/sensor-imputation-thesis/src/sensor_imputation_thesis/nadire/9missingcolsdata.parquet")

df.describe()

#All preprocessing steps(drop columns with different indexes)
df1=df.drop(columns=["te_exh_cyl_out__1","te_exh_cyl_out__2","te_exh_cyl_out__3","te_exh_cyl_out__4",
             "te_exh_cyl_out__5","te_exh_cyl_out__6"])
#filter df with engine running
df1=df1[df1["fr_eng"]>0]

#Check nan values in each column
for col in df1.columns:
    print(f"Column {col} has {df[col].isna().sum()} NaN values")

#Assign dummy values to nan columns 
## For target column ("the y"), assign single dummy value for the whole column (all rows);
## For columns with partial nan values, assign one same dummy value (out of the min_max scope) for nan values;
## For columns with all rows' nan values, assign one dummy (-1) for the column.
# Redefine imputation functions

def impute_nan(data, target_col, scale=1.0, epsilon=1e-9):
    """
    Replace NaN values with dummy values based on specified criteria.

    Args:
       data: Input DataFrame
       target_col: The target column name that needs a single dummy value for the entire column
       scale: Controls how far outside the min/max the imputed values are placed for non-target columns.

    Returns:
      Data with NaNs replaced by dummy values
    """
    df=data.copy()

    for col in df.columns:
        #Assign one single dummy to the whole target_col
        if target_col in df.columns:
            df[target_col] = -2

        #if no nan, skip
        if df[col].isna().sum() == 0:
            continue

        #same dummy for the col with all nan values 
        if df[col].isna().sum() == len(df):
            df[col]=-1
        
        if df[col].isna().sum() > 0:
            # Columns with partial NaN values, assign dummy value outside the min/max range
            min_val = df[col].min(skipna=True)
            max_val = df[col].max(skipna=True)

            # Edge case: all values are identical (min == max)
            if min_val == max_val:
                min_val -= epsilon
                max_val += epsilon

            # Generate a dummy value strictly outside the min/max range
            dummy_value = max_val + scale

            # Impute NaN values with the dummy value
            df.loc[df[col].isna(), col] = dummy_value

    return df
            
        
len(df1) 



import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import argparse
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
import mlflow
import mlflow.pytorch

#This is the overall approach, but somehow it does not show in mlflow. 
# And I also need to apply gridsearchcv 
# Change the format of plot since the current one is not clear (also consider only leaving the column)

# Select numeric columns
numeric_df = df1.select_dtypes(include='number')

# Standardize the numeric data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_df)

# Convert scaled data back to DataFrame
scaled_df = pd.DataFrame(scaled_data, columns=numeric_df.columns)

# Impute NaN values
data = impute_nan(scaled_df,"te_exh_cyl_out__0")

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

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# Configuration parameters (This part should be defined in the same cell when running in Jupyter)
class Config:
    batch_size = 128
    epochs = 10
    no_cuda = False
    no_mps = False
    seed = 1
    log_interval = 10

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

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD

#Initialize Mlflow
mlflow.set_experiment("VAE Training")

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device).float()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"NaN or Inf detected in loss at batch {batch_idx}.")
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device).float()  # Convert to FloatTensor
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

#Initiate mlflow autolog 
run_name="vae_for_te_exh_cyl_out__0"
mlflow.pytorch.autolog()

if __name__ == "__main__":
  with mlflow.start_run(run_name=run_name):
        for epoch in range(1, args.epochs + 1):
            train_loss = train(epoch)
            test_loss = test(epoch)

            with torch.no_grad():
                # Generate a latent sample
                sample = torch.randn(64, 20).to(device)  # Adjust latent dimensions
                # Decode the sample
                sample = model.decode(sample).cpu().numpy()  
                # Validate the shape of the decoded sample
                if sample.shape[1] != len(numeric_df.columns):
                    raise ValueError(
                        f"Shape mismatch: Generated data has {sample.shape[1]} columns, "
                        f"but expected {len(numeric_df.columns)}."
                    )

                # Reverse the scaling to map back to the original distribution
                generated_df = pd.DataFrame(scaler.inverse_transform(sample), columns=numeric_df.columns)
                print(generated_df.head())

# Compare stats
def compare_stats(numeric_df, generated_df):
    stats={}
    stats['mean_real']=np.mean(numeric_df,axis=0)
    stats['mean_synthetic']=np.mean(generated_df,axis=0)
    stats['mean_diff']=(stats['mean_synthetic']-stats['mean_real'])/stats['mean_real']
    stats['std_real']=np.std(numeric_df,axis=0)
    stats['std_synthetic']=np.std(generated_df,axis=0)
    stats['std_diff']=(stats['std_synthetic']-stats['std_real'])/stats['std_real']
    return stats

stats=compare_stats(numeric_df,generated_df)
print("Stats Comparison:",stats)

# Plot histograms
for col in numeric_df.columns:
    plt.figure()
    numeric_df[col].plot(kind='hist', alpha=0.5, label='Real')
    generated_df[col].plot(kind='hist', alpha=0.5, label='Generated')
    plt.legend()
    plt.title(f'Distribution of {col}')
    plt.show()
