#Import Pypots Library
from pypots.optim import Adam
from pypots.imputation import SAITS
#from pypots.utils.metrics import calc_mae
from pypots.nn.functional import calc_mae


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

torch.cuda.empty_cache()
#PatchTST might be an ideal choise if SAITS is too slow 

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
start, stop = pd.Timestamp("2023-10-01"), pd.Timestamp("2024-02-01")

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
    "se_mip_acco__0",
    "se_mip_acco__1",
    "se_mip_acco__2",
    "se_mip_acco__3",
    "se_mip_acco__4",
    "se_mip_acco__5",
    "fr_eng_ecs",
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
    "te_air_scav_rec",
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
product_id = "6d48e8c49e3ebecad7aef2f1e069ec53"

#product_id 
#add more product id with same engine type (fine the most popular ones)


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


#save df to parquet 




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

# filter df with engine running (changed into 10/60 revolutions)
df1 = df[df["fr_eng"] > (10/60)]

file_path="/home/ec2-user/SageMaker/sensor-imputation-thesis/src/sensor_imputation_thesis/nadire/dataframeforpypots"

df1.to_parquet(f'{file_path}.parquet')

# Check nan values in each column
for col in df1.columns:
    print(f"Column {col} has {df1[col].isna().sum()} NaN values")
    missing_rate=df1[col].isna().sum()/len(df1[col])
    print(f"Column {col} has {missing_rate} Missing_rate")

#len(df1)


## Attempt with CAITS
#split dataset 
sensor_cols=[col for col in df1.columns if col!="time"]
data=df1[sensor_cols].values

## Convert data to 3D arrays of shape n_samples, n_timesteps, n_features, X_ori refers to the original data without missing values 
## Reconstruct all columns simultaneously  #num_features: 119, num_samples:119
n_features=118 #exclude the time column
n_steps=1440 #window length, 1440 steps=24 hours of 1-minute data
n_samples=data.shape[0]// n_steps
n_features=data.shape[1]

train_X=data[:n_samples*n_steps].reshape(n_samples,n_steps,n_features)

print(train_X)


#Split into train, test, val

train_size = int(0.6 * len(train_X))
val_size = int(0.2 * len(train_X))
test_size = len(train_X) - train_size - val_size

train_X_split = train_X[:train_size]
val_X = train_X[train_size:train_size + val_size]
test_X = train_X[train_size + val_size:]
 


#Optional: Artificially mask? 
#from pypots.utils import mcar
#val_X_ori=val_X.copy()
#val_X_missing=mcar(val_X_ori,0.3)
#val_X=val_X_missing


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

if args.cuda:
    device = torch.device("cuda")
    print("Using CUDA")
elif use_mps:
    device = torch.device("mps")
    print("Using MPS")
else:
    device = torch.device("cpu")
    print("Using CPU")


# initialize the model (from example)
saits = SAITS(
    n_steps=train_X.shape[1],
    n_features=train_X.shape[2],
    n_layers=2, #deep network(4) for long sequences, here is revised for 2 since it always shows cuda error 
    d_model=512,  #d_modle must equal n_heads * d_k
    d_ffn=512,
    n_heads=8,
    d_k=64,
    d_v=64,
    dropout=0.1,
    attn_dropout=0.1,
    diagonal_attention_mask=True,  # otherwise the original self-attention mechanism will be applied
    ORT_weight=1,  # you can adjust the weight values of arguments ORT_weight
    # and MIT_weight to make the SAITS model focus more on one task. Usually you can just leave them to the default values, i.e. 1.
    MIT_weight=1,
    batch_size=8,
    # here we set epochs=10 for a quick demo, you can set it to 100 or more for better performance
    epochs=5,
    # here we set patience=3 to early stop the training if the evaluting loss doesn't decrease for 3 epoches.
    # You can leave it to defualt as None to disable early stopping.
    patience=3,
    # give the optimizer. Different from torch.optim.Optimizer, you don't have to specify model's parameters when
    # initializing pypots.optim.Optimizer. You can also leave it to default. It will initilize an Adam optimizer with lr=0.001.
    optimizer=Adam(lr=1e-3),
    # this num_workers argument is for torch.utils.data.Dataloader. It's the number of subprocesses to use for data loading.
    # Leaving it to default as 0 means data loading will be in the main process, i.e. there won't be subprocesses.
    # You can increase it to >1 if you think your dataloading is a bottleneck to your model training speed
    num_workers=0,
    # just leave it to default as None, PyPOTS will automatically assign the best device for you.
    # Set it as 'cpu' if you don't have CUDA devices. You can also set it to 'cuda:0' or 'cuda:1' if you have multiple CUDA devices, even parallelly on ['cuda:0', 'cuda:1']
    device=device, 
    # set the path for saving tensorboard and trained model files 
    saving_path="/home/ec2-user/SageMaker/sensor-imputation-thesis/src/sensor_imputation_thesis/nadire/best_model",
    # only save the best model after training finished.
    # You can also set it as "better" to save models performing better ever during training.

    model_saving_strategy="best",
)




# train the model on the training set, and validate it on the validating set to select the best model for testing in the next step
saits.fit(train_set={"X": train_X}, val_set={"X": val_X, "X_ori":val_X})


# the testing stage, impute the originally-missing values and artificially-missing values in the test set
saits_results = saits.predict(test_X)
saits_imputation = saits_results["imputation"]

#clear cuda after prediction if on cuda 

if args.cuda:
    torch.cuda.empty_cache() 

# calculate mean absolute error on the ground truth (artificially-missing values)
testing_mae = calc_mae(
    saits_imputation, 
    {"test_X":test_X},
    {"test_ori":test_X}
    
)
print(f"Testing mean absolute error: {testing_mae:.4f}")


##Current error is on the cuda 
#test memory
def test_memory(in_size=100, out_size=10,hidden_size=100,optimizer_type=torch.optim.Adam, batch_size=1, device=0):
    sample_input=torch.randn(batch_size,in_size)
    model=saits,
    optimizer=optimizer_type(model.parameters(),lr=1e-3)
    print("Beginning mem:",torch.cuda.memory_allocated(device))
    model.to(device)
    print()
    for i in range(3):
        print("Iteration",i)
        a=torch.cuda.memory_allocated(device)
        out=model(sample_input.to(device)).sum()
        b=torch.cuda.memory_allocated(device)
        print("2-After forward pass", torch.cuda.memory_allocated(device))
        print("2-Memory consumed by forward pass", b-a)
        out.backward()
        print("3-After backward pass", torch.cuda.memory_allocated(device))
        optimizer.step()
        print("4-After optimizer step", torch.cuda.memory_allocated(device))