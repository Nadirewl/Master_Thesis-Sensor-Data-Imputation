#Comparing with the vae 2, there are 2 updates:
# 1. imputation method is changed into assigning -2 to all nan values (including the target column, columns with all nan 
#    and partial nan);
# 2. loss function is changed, current method is to calculate the mse for target column (the one wholy masked) 
## Currently discarded approach: nanified (cols that are imputed during preprossing) and turn it into np array/vector. 
# 3. Scaler is changed into min_max scaler 
# 4. Batch size changed to 64


import sensor_imputation_thesis.shared.load_data as load 
from data_insight import setup_duckdb
from duckdb import DuckDBPyConnection as DuckDB
import pandas as pd
from duckdb import DuckDBPyRelation as Relation
from pathlib import Path
import hashlib
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import argparse
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
import mlflow
import mlflow.pytorch


#Read data presaved
df=pd.read_parquet("/home/ec2-user/SageMaker/sensor-imputation-thesis/src/sensor_imputation_thesis/nadire/9missingcolsdata.parquet")

#All preprocessing steps(drop columns with different indexes)
df1=df.drop(columns=["te_exh_cyl_out__1","te_exh_cyl_out__2","te_exh_cyl_out__3","te_exh_cyl_out__4",
             "te_exh_cyl_out__5","te_exh_cyl_out__6"])
#filter df with engine running
df1=df1[df1["fr_eng"]>0]

#Check nan values in each column
for col in df1.columns:
    print(f"Column {col} has {df[col].isna().sum()} NaN values")

#Updated Function: Assign dummy values to nan columns/values 

def impute_nan(data, target_col):
    """
    Replace NaN values with dummy values based on specified criteria.

    Args:
       data: Input DataFrame
       target_col: The target column name that needs a single dummy value for the entire column.

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

        #same dummy for the col with nan values (all & partial)    
        if df[col].isna().sum() > 0:
            # Columns with partial NaN values, assign dummy value -2 
            df.loc[df[col].isna(), col] = -2

    return df
            

#This is the overall approach, but somehow it does not show in mlflow. 
# And I also need to apply gridsearchcv 
# Change the format of plot since the current one is not clear (also consider only leaving the column)

# Select numeric columns
numeric_df = df1.select_dtypes(include='number')

# Standardize the numeric data
scaler = MinMaxScaler() 
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
    batch_size = 64
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

# Specify the target column
targetcolumn = 'te_exh_cyl_out__0'
targetcol_idx = data.columns.get_loc(targetcolumn)


# Reconstruction + KL divergence losses summed over target column and batch
def loss_function(recon_x, x, mu, logvar,targetcol_idx):
    # Specify the target column
    recon_x_col=recon_x[:,targetcol_idx]
    x_col=x[:,targetcol_idx]

    MSE = F.mse_loss(recon_x_col, x_col, reduction='sum')
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
        loss = loss_function(recon_batch, data, mu, logvar,targetcol_idx)
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
        average_train_loss=train_loss/len(train_loader.dataset)
    print('====> Epoch: {} Average train loss: {:.4f}'.format(epoch, average_train_loss))
    return average_train_loss
      
  
                
def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device).float()  # Convert to FloatTensor
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar,targetcol_idx).item()
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    average_test_loss=test_loss / len(test_loader.dataset)
    print('====> Epoch: {} Average test loss: {:.4f}'.format(epoch, average_test_loss))
    return average_test_loss


#Initiate mlflow autolog 
run_name="vae_for_te_exh_cyl_out__0"
mlflow.pytorch.autolog()
input_example=df1.iloc[:1]

if __name__ == "__main__":
  with mlflow.start_run(run_name=run_name):
        for epoch in range(1, args.epochs + 1):
            average_train_loss = train(epoch)
            average_test_loss = test(epoch)
           
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

            #log mlflow metric
                mlflow.log_metric("average_train_loss", average_train_loss, step=epoch)
                mlflow.log_metric("average_test_loss", average_test_loss, step=epoch)
                mlflow.log_metric("epoch_num:", 10)
                mlflow.log_metric("learning_rate:",1e-4)



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

stats=compare_stats(numeric_df[targetcolumn],generated_df[targetcolumn])
print("Stats Comparison:",stats)

#Density plot

# Filter the original and imputed data for the specified column
orig_col_filtered = numeric_df[targetcolumn].dropna()
imputed_col_filtered = generated_df[targetcolumn].dropna()

# Create the density plot
plt.figure(figsize=(10, 6))
sns.kdeplot(orig_col_filtered, fill=True, label='Original')
sns.kdeplot(imputed_col_filtered, fill=True, label='Imputed')
plt.title(f'Density Plot of {targetcolumn}')
plt.legend()
plt.tight_layout()
plt.show()


