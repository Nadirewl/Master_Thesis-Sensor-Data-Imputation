import os

import data_insight
import numpy as np
import pandas as pd
from uc90core.performance_index.perf_idx import PerformanceIndex

from sensor_imputation_thesis.shared.baba_wrapper import BabayagaWrapper

bbconfig = {
    "PROFILE": "production",
    "babayaga_system": "S3",
    "babayaga_config_name": "2-Stroke/Reference-Data/Configuration/ref_curve_config.yaml",
    "babayaga_config_location": "prod-analytics-hydra-api--hydra",
}
os.environ.update(bbconfig)


def generate_dummy_data(n_cyl: int, n_tc: int) -> pd.DataFrame:
    def index(tag: str, n_cols: int, value: float) -> list[str]:
        return {f"{tag}_{i}": value for i in range(n_cols)}

    required_opvs = {
        "in_reference_mode": 1,
        "fr_eng_ecs": 1,
        "in_stable": 1,
        "pr_air_scav_iso": np.nan,
        "pr_cyl_comp_mv_iso": np.nan,
        "pr_cyl_max_mv_iso": np.nan,
        # "pr_air_scav_iso": 1,
        # "pr_cyl_comp_mv_iso": 1,
        # "pr_cyl_max_mv_iso": 1,
        "re_eng_load_ecs": 1,
        "re_eng_load_pmi": 1,
        "te_cw_ic_in_common": 1,
        # CYL values
        **index("pr_cyl_comp_", n_cyl, 1e5),
        **index("pr_cyl_max_", n_cyl, 1e5),
        **index("se_mip_", n_cyl, 1e5),
        # TC values
        **index("te_air_ic_out_", n_tc, 1e5),
        **index("fr_tc_iso_", n_tc, 1e5),
        **index("te_exh_turb_in_iso_", n_tc, 1e5),
    }

    data = pd.DataFrame([required_opvs])
    return data


data = generate_dummy_data(n_cyl=6, n_tc=2)
# data.to_parquet('my_data.parquet')
# pd.read_parquet('my_data.parquet')

con = data_insight.setup_duckdb()
con.sql("SET enable_progress_bar=true")

# I picked this one because it exists both in "timeseries" and "shipinfo"
pid = "237848bc2ed636c1ac85baea4755344c"
engine_type = con.sql(f"SELECT engine_type FROM shipinfo WHERE productId == '{pid}'").df().engine_type.item()
engine_uuid = con.sql(f"SELECT uuid FROM site_infos WHERE product_id == '{pid}'").df().uuid.item()

# If we have multiple reference modes, we cannot evaluate them all at once
# We could also just assume that reference_mode is 1
[reference_mode] = data["in_reference_mode"].unique()
reference_mode = int(reference_mode)
# reference_mode = 1

# con.sql("select in_stable from timeseries where in_stable is not null limit 1")
con.sql("select * from timeseries where in_stable is not null and in_reference_mode is not null limit 1")

engine_uuid = "72c9e0a1-17f5-4397-8e45-8433dea65883"


babayaga_wrapper = BabayagaWrapper(engine_uuid=engine_uuid, reference_mode=reference_mode)

model = PerformanceIndex(
    fr_eng_mcr=babayaga_wrapper.fr_eng_smcr,
    **babayaga_wrapper.reference_curves,
    reference_mode=reference_mode,
    engine_type=engine_type,
)
result = model.evaluate(data)
result
