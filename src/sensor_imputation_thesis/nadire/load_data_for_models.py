import pandas as pd
from duckdb import DuckDBPyConnection as DuckDB
from duckdb import DuckDBPyRelation as Relation
import data_insight
from data_insight import setup_duckdb
from duckdb import DuckDBPyRelation as Relation
from pathlib import Path
import hashlib
from duckdb import DuckDBPyConnection as DuckDB
from sklearn.preprocessing import OneHotEncoder

# Load data
pd.set_option("display.max_columns", None)

def load_engine_data(
    con: DuckDB, product_id: str, start: pd.Timestamp, stop: pd.Timestamp, tags: list[str]
) -> Relation:
    return con.sql(f"""
    SELECT {','.join(tags)}
    FROM timeseries
    WHERE
        in_stable IS NOT NULL
        AND in_reference_mode IS NOT NULL
        AND time BETWEEN '{start}' AND '{stop}'
        AND pid = '{product_id}'
    """)

def get_tags_hash(tags):
    return hashlib.md5(",".join(tags).encode()).hexdigest()

# edit timestamp as needed
start, stop = pd.Timestamp("2019-10-01"), pd.Timestamp("2024-10-01")

#This tag list combines all the variables needed for calculating sfoc and features that influence the variation of variables 
tags = [
    "time",
    "bo_aux_blower_running", #1 
    "bo_gov_mode_rpm",
    "in_engine_running_mode", #1
    "in_reference_mode",
    "in_stable",
    "pd_air_ic__0 as pd_air_ic", #1 
    "pr_air_scav_ecs", #1
    "pr_baro", #1
    "pr_cyl_comp_mv", #add 
    "pr_cyl_comp_mv_iso",
    "pr_cyl_max_mv", #add
    "pr_cyl_max_mv_iso", #1
    "pr_exh_rec", #1
    "pr_exh_turb_out__0 as pr_exh_turb_out", #1
    "pr_pcomp_acco__0 as pr_pcomp_acco", #add large missing number
    "pr_pcomp_ordered", #add not missing
    "pr_pcomp_ordered_acco__0 as pr_pcomp_ordered_acco", #add, large missing number
    "pr_cyl_max__0 as pr_pmax", #1
    "pr_pmax_acco__0 as pr_pmax_acco",  #add large missing number 
    "pr_pmax_ordered", #add not missing
    "pr_pmax_ordered_acco__0 as pr_pmax_ordered_acco", #add, large missing number 
    "pr_prise_acco__0 as pr_prise_acco", #add, large missing number 
    "pr_prise_ordered_acco__0 as pr_prise_ordered_acco", #add, large missing number 
    "re_eng_load", #1
    "re_eng_load_estimate_ecs", #add not missing a lot
    "re_perf_idx_hrn_indicator as re_perf360_hrn_indicator", #add 98% percent mising values 
    # "tc_turbocharger_efficiency",
    "te_air_comp_in_a__0 as te_air_comp_in_a", #1
    "te_air_scav_rec_iso as te_air_scav_iso", #1
    "te_air_scav_rec", #1
    "te_exh_turb_in__0 as te_exh_turb_in", #1
    "te_seawater", #1
    #the rest are those from xg boost have higher feature importance
     "fr_eng", 
     "te_exh_cyl_out__0", 
     "pd_air_ic__0",
     "te_air_ic_out__0", 
     "te_air_comp_in_b__0",
     "fr_tc__0", 
     "te_exh_turb_out__0",
     "pr_air_scav", 
     "pr_cyl_comp__0", 
     "se_mip__0", 
     "fr_eng_setpoint", 
     "pr_cyl_comp_mv_iso", 
     "fr_eng_ecs",
     "pr_air_scav_iso",

]


product_id = "ea52565f40ed312a2f3a18071998ce0a"  #From dataframe specialized for engine type ME-LGIM

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
print(len(df))


## Adding engine types and assign onehot encoder, and merge df  
con = data_insight.setup_duckdb()
con.sql("SET enable_progress_bar=true")

engine_type = con.sql(f"SELECT engine_type FROM shipinfo WHERE productId = '{product_id}'").df().engine_type.iloc[0]

#engine_type = con.sql(f"SELECT engine_type FROM shipinfo WHERE productId == '{product_id}'").df().engine_type.item()

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

file_path="/home/ec2-user/SageMaker/sensor-imputation-thesis/src/sensor_imputation_thesis/nadire/dffinal_for_comparison"

df1.to_parquet(f'{file_path}.parquet')




##only keep indices of 0, 9 missing columns and the corresponding columns with higher rank in the feature importance 
tags=[
    #9 main missing columns
    "time",
    "fr_eng",
    "te_exh_cyl_out__0",
    "pd_air_ic__0",
    "pr_exh_turb_out__0",
    "te_air_ic_out__0",
    "te_seawater",
    "te_air_comp_in_a__0",
    "te_air_comp_in_b__0",
    "fr_tc__0",
    "pr_baro",
    #most common "bad"/ "out of bound" sensors
    "pd_air_ic__0",  #overlapped with below
    "pr_exh_rec",
    "te_exh_turb_in__0",
    "te_exh_turb_out__0",
    #Feature Importance Result
    #for te_exh_cyl_out__0
    "bo_aux_blower_running",
    "re_eng_load",
    "pr_air_scav_ecs", #overlap
    #pd_air_ic__0
    "pr_air_scav",
    #for te_air_ic_out__0
    #for te_seawater
    "te_air_scav_rec",
    "te_air_ic_out__0",
    "pr_cyl_comp__0",
    "pr_cyl_max__0",
    "se_mip__0",
    "te_exh_cyl_out__0",
    "fr_eng_setpoint",
    "te_air_scav_rec_iso",
    #for pr_baro
    "pr_cyl_max_mv_iso",
    "pr_cyl_comp_mv_iso",
    "fr_eng_ecs",
    "pr_air_scav_iso",

]
