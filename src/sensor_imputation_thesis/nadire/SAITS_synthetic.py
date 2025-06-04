import numpy as np
import pandas as pd
from pygrinder.missing_completely_at_random import mcar
from sklearn.preprocessing import MinMaxScaler
from pypots.imputation import SAITS
from pypots.nn.functional import calc_mae
from pypots.optim import Adam

df=pd.read_parquet("/home/ec2-user/SageMaker/sensor-imputation-thesis/src/sensor_imputation_thesis/nadire/Newdataframeforpypots.parquet")
df=df[:20000]

#Try to drop all nan cols and rows 
df=df.dropna(axis=1,how='all')
df=df.dropna(axis=0,how='any')

sensor_cols = [col for col in df.columns if col != "time"]
feature_names=df[sensor_cols].columns.tolist()
data = df[sensor_cols].values

n_features = data.shape[1]  # exclude the time column
n_steps = 30 #60 (was 60 previously) #(TRY TO CHANGE HERE)  # # window length, 1440 steps = 24 hours of 1-minute data, but here is revised to 60 again
#total_elements = data.shape[0] * data.shape[1]
n_samples = data.shape[0] // n_steps 

data_reshaped=data[:n_samples*n_steps].reshape(n_samples,n_steps,n_features)
print(f"Reshaped data:{data.shape}")


scalers={}
data_normalized=data.copy()

data_normalized = np.zeros_like(data_reshaped)  # Initialize the normalized data array
scalers = {}

for i in range(data_reshaped.shape[2]):
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Flatten timesteps and samples for scaling
    data_normalized[:, :, i] = scaler.fit_transform(data_reshaped[:, :, i].reshape(-1, 1)).reshape(data_reshaped.shape[0], data_reshaped.shape[1])
    scalers[i] = scaler  # Save scalers to inverse-transform later


def mcar_f(X, mask_ratio=0.2):
    """Apply MCAR only to observed values."""
    observed_mask=~np.isnan(X) #find observed positions
    artificial_mask=mcar(X,mask_ratio).astype(bool) #generate MCAR mask, cast to boolean
    #combine masks 
    combined_mask=observed_mask & artificial_mask

    #Apply masking
    X_masked=X.copy()
    X_masked[combined_mask]=np.nan
    return X_masked,combined_mask

artificially_masked_data, mask =mcar_f(data_normalized)
data_ori=data_normalized.copy() #Ground truth

SAITS=SAITS(n_steps=1440, 
            n_features=data.shape[1],
            n_layers=2,
            d_model=512,
            d_ffn=512,
            n_heads=8,
            d_k=64,
            d_v=64,
            dropout=0.1,
            attn_dropout=0.1,
            diagonal_attention_mask=True,
            ORT_weight=1,
            MIT_weight=1,
            batch_size=5,
            epochs=10, 
            patience=6,
            optimizer=Adam(lr=1e-3),
            num_workers=0,
            saving_path="/home/ec2-user/SageMaker/sensor-imputation-thesis/src/sensor_imputation_thesis/nadire/best_model",
            model_saving_strategy="best",
            device="cpu")

artificially_masked_data_dict = {"X": artificially_masked_data}

SAITS.fit(artificially_masked_data_dict)
imputed_data=SAITS.impute(artificially_masked_data)

def inverse_scale(imputation, scalers):
    n_features = imputation.shape[2]
    imputation_denorm = np.empty_like(imputation)
    
    for i in range(n_features):
        imputation_denorm[:, :, i] = scalers[i].inverse_transform(imputation[:, :, i].reshape(-1, 1)).reshape(imputation.shape[0], imputation.shape[1])
    
    return imputation_denorm  # Move the return statement outside the loop

    
#Apply function to the dataset 
imputation_denorm=inverse_scale(imputed_data,scalers)
ori_denorm=data_reshaped

mae_per_feature=[]
for i in range(n_features):
    #Extract imputation and ground truth for feature i
    imputation_i=imputation_denorm[:,:,i]
    ground_truth_i=ori_denorm[:,:,i]
    mask_i=mask[:,:,i]
    # Check for NaN values
    if np.isnan(imputation_i).any() or np.isnan(ground_truth_i).any():
        print(f"NaN values detected in feature {i}")
        continue  # Skip this feature if NaN values are found
    #Filter only artificially masked positions
    mae_i=calc_mae(imputation_i,ground_truth_i,mask_i)
    mae_per_feature.append(mae_i)
    print(f"MAE for {feature_names[i]}: {mae_i:.4f}")
  