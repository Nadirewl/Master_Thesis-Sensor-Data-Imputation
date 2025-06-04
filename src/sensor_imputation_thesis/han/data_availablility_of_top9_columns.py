# this python file aims to investigate the data availability of top 9 columns
# lacking most information for all the engines, output the missing value percentage
# named database_data_availability

import functools
import hashlib
from pathlib import Path

import data_insight
import pandas as pd
from data_insight import setup_duckdb
from duckdb import DuckDBPyConnection as DuckDB
from duckdb import DuckDBPyRelation as Relation

pd.set_option("display.max_columns", None)

con = data_insight.setup_duckdb()

DATA = Path.home() / "SageMaker"
DATA.mkdir(exist_ok=True)
tags = [
    "time",
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
    "te_air_comp_in_b__0",
    "te_air_comp_in_b__1",
    "te_air_comp_in_b__2",
    "te_air_comp_in_b__3",
    "fr_tc__0",
    "fr_tc__1",
    "fr_tc__2",
    "fr_tc__3",
    "pr_baro",
    "pr_exh_rec",
    "pr_air_scav_ecs",
    "te_air_scav_rec",
    "te_exh_turb_in__0",
    "te_exh_turb_in__1",
    "te_exh_turb_in__2",
    "te_exh_turb_in__3",
    "re_eng_load",
]


def count_missing_values(df, product_id):
    total_values = len(df)
    if total_values == 0:
        print(f"No data availability for product_id: {product_id}")
        return None
    else:
        missing_values = df.isna().sum()
        missing_percent = (missing_values / total_values) * 100
        
        # Create a DataFrame with product_id as index and their corresponding missing percent as columns
        missing_percent_df = pd.DataFrame(missing_percent).transpose()
        missing_percent_df['product_id'] = product_id
        
        return missing_percent_df



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

# Find the overlapping productid values
overlapped_id = con.sql("""
    SELECT s.productid
    FROM (SELECT DISTINCT productid FROM shipinfo) s
    JOIN (SELECT DISTINCT pid FROM timeseries) t ON s.productid = t.pid
""").fetchall()

# Convert the result to a list
product_id_list = [row[0] for row in overlapped_id]


start, stop = pd.Timestamp("2024-09-01"), pd.Timestamp("2024-09-30")

# Initialize an empty DataFrame to store all missing values
all_missing_values_df = pd.DataFrame()
for product_id in product_id_list:
    cache = DATA / f"data_{get_tags_hash(tags)}_{product_id}_{start:%Y-%m-%d}_{stop:%Y-%m-%d}.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
    else:
        df = load_engine_data(setup_duckdb(), product_id, start, stop, tags).df()
        df.to_parquet(cache)

    # Count missing values and get the DataFrame for each product_id
    missing_values_df = count_missing_values(df, product_id)
    
    if missing_values_df is not None:
        # Get engine type for the product_id
        engine_type_query = con.sql(f"SELECT engine_type FROM shipinfo WHERE productId = '{product_id}'").df()
        if len(engine_type_query) == 1:
            engine_type = engine_type_query.engine_type.item()
        elif len(engine_type_query) > 1:
            engine_type = engine_type_query.engine_type.iloc[0]  # Take the first value or handle as needed
        else:
            engine_type = 'Unknown'  # Assign a default value if engine type is not found
        
        print(f"Product ID: {product_id}, Engine Type: {engine_type}")
        print(f"Missing Values DataFrame:\n{missing_values_df}")
        
        missing_values_df['engine_type'] = engine_type
        
        all_missing_values_df = pd.concat([all_missing_values_df, missing_values_df], ignore_index=True)
    else:
        print(f"Missing values DataFrame is None for product_id: {product_id}")

# Reorder columns to have product_id and engine_type first
all_missing_values_df = all_missing_values_df[
    ["product_id", "engine_type"]
    + [col for col in all_missing_values_df.columns if col not in ["product_id", "engine_type"]]
]

path = DATA / "sensor-imputation-thesis/src/sensor_imputation_thesis/han/database_data_availability_202409sfoc"
path.parent.mkdir(exist_ok=True, parents=True)
all_missing_values_df.to_parquet(path, index=False)
