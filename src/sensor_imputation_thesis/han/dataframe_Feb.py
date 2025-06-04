# create the dataframe
from data_insight import setup_duckdb
from duckdb import DuckDBPyConnection as DuckDB
import pandas as pd
from duckdb import DuckDBPyRelation as Relation
from pathlib import Path
import hashlib

pd.set_option('display.max_columns', None)

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
    return hashlib.md5(','.join(tags).encode()).hexdigest()

start, stop = pd.Timestamp("2023-11-01"), pd.Timestamp("2024-10-31")
tags = [
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
    "te_exh_turb_in__0",
    "te_exh_turb_out__0",
    "pr_exh_rec",
    "pr_air_scav",
    "pr_air_scav_ecs",
    "fr_eng_setpoint",
    "te_air_ic_out__0",
    "pr_cyl_max__0",
    "se_mip_acco__0",
    "fr_eng_ecs",
    "se_mip__0",
    "pr_cyl_comp__0",
    "te_cw_ic_in_common",
    "te_air_ic_out__0",
    "in_stable",
    "te_exh_turb_in_iso__0",
    "fr_tc_iso__0",
    "pr_cyl_max_mv_iso",
    "pr_cyl_comp_mv_iso",
    "pr_air_scav_iso",
    "te_air_scav_rec",
    "te_air_scav_rec_iso",
    "re_perf_idx_hrn_indicator",
    "in_engine_running_mode",
    "te_exh_turb_in__0",
    "bo_aux_blower_running",
    "re_eng_load",
    ]
product_id = "d6a77f306e013a579cb4eb3a7ed3571b"

cache = Path(f'/tmp/data_{get_tags_hash(tags)}_{product_id}.parquet')
if cache.exists():
    df = pd.read_parquet(cache)
else:
    con = setup_duckdb()
    df = load_engine_data(con, product_id, start, stop, tags).df()
    df.to_parquet(cache)



path = "/home/ec2-user/SageMaker/sensor-imputation-thesis/src/sensor_imputation_thesis/han/dataframe_forxgboost"
df.to_parquet(path, index=False)


