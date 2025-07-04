import functools
import hashlib
import os
from pathlib import Path

import data_insight
import pandas as pd
from babayaga.clients.baba_yaga_client import BabaYagaClient
from codebase_io.analytics_data.cached_analytics_data_client import CachedAnalyticsDataClient as ADC
from data_insight import setup_duckdb
from duckdb import DuckDBPyConnection as DuckDB
from duckdb import DuckDBPyRelation as Relation
from uc286core.uc220core_360.sfoc import ExcessSFOC
from util2s.moodel.ceon_assets import CeonEngine
from util2s.reference_mode.reference_mode_selector import ReferenceModeSelector

from sensor_imputation_thesis.shared.baba_wrapper import BabayagaWrapper

bbconfig = {
    "PROFILE": "production",
    "babayaga_system": "S3",
    "babayaga_config_name": "2-Stroke/Reference-Data/Configuration/ref_curve_config.yaml",
    "babayaga_config_location": "prod-analytics-hydra-api--hydra",
    "USE_CACHE": "true",
}
os.environ.update(bbconfig)


adc = ADC("production")
con = data_insight.setup_duckdb()
con.sql("SET enable_progress_bar=true")

# I picked this one because it exists both in "timeseries" and "shipinfo"
# pid = "237848bc2ed636c1ac85baea4755344c"
pid = "ea52565f40ed312a2f3a18071998ce0a"
engine_type = con.sql(f"SELECT engine_type FROM shipinfo WHERE productId == '{pid}'").df().engine_type.item()
engine_uuid = con.sql(f"SELECT uuid FROM site_infos WHERE product_id == '{pid}'").df().uuid.item()
templates = adc.find_services_for_service_template("c1ce91e3-ac23-4153-bcf3-29aae039dba4")
services = adc.find_services(asset_uuid=engine_uuid, asset_traverse=False)
service_uuid = services[services.apply(lambda x: x.config.get("app") == "uc00286-perf360", axis=1)].uuid

df = pd.read_parquet(
    "/home/ec2-user/SageMaker/sensor-imputation-thesis/src/sensor_imputation_thesis/nadire/df_combined.parquet"
)


@functools.cache
def baba(engine_uuid: str) -> BabayagaWrapper:
    return BabaYagaClient(engine_uuid)


result = []
for hour, group_df in df.set_index("time").groupby(pd.Grouper(freq="h")):
    print(f"Processing hour: {hour}")
    try:
        assert group_df["in_stable"].eq(1).all()
        [reference_mode] = group_df["in_reference_mode"].unique()
    except (AssertionError, ValueError):
        continue  # Not stable or no unique reference mode for this hour, skip it

    hourly_df = group_df.mean().to_frame().T
    required_ref_data = ["te_air_scav_iso", "pr_cyl_max_mv_iso", "mr_fuel_iso", "tc_eff", "pr_exh_turb_out_iso"]
    ref_data = pd.concat(
        [
            ReferenceModeSelector(engine_uuid, baba(engine_uuid)).get_reference_dataframe(df_input=hourly_df),
            pd.DataFrame(data=dict.fromkeys(required_ref_data, [])),
        ]
    )
    engine_characteristics = {
        "mcr_power_kw": baba(engine_uuid).get_metadata().mcr_power_kw,
        "engine_tuning": CeonEngine(engine_uuid).neo4j_attr.tuning_strategy,
    }
    esfoc = ExcessSFOC(ref_data=ref_data, engine_type=engine_type, engine_characteristics=engine_characteristics)
    result.append(dict(esfoc.evaluate(hourly_df).squeeze().items()))

df = pd.DataFrame(result).dropna(how="all", axis=1)
# df.to_parquet("/home/ec2-user/SageMaker/sensor-imputation-thesis/src/sensor_imputation_thesis/nadire/sfoc_afterimpute.parquet")
# print(df)
