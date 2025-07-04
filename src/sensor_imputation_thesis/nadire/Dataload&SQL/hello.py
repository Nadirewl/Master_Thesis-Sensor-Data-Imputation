#import sensor_imputation_thesis
import sensor_imputation_thesis.shared.load_data as load 
from data_insight import setup_duckdb
from duckdb import DuckDBPyConnection as DuckDB
import pandas as pd
from duckdb import DuckDBPyRelation as Relation
#def load_data con: DuckDB, product_id: str, tags: list[str]

con = setup_duckdb()


r=con.sql("""
    SELECT emission_reduction 
    FROM engines
    WHERE emission_reduction NOT NULL
    LIMIT 100
    """)
    

print(r)


e=con.sql("""
        SELECT *
        FROM engines
          """)
print(e)

start, stop = pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-01")
tags = [
        "time",
        "fr_eng",
        "an_phase_setpoint",
        "bo_gov_mode_index",
        "pd_cw_liner",
        "pr_exh_turb_out",
    ]
product_id = "4379247-1"
    
load.load_engine_data(con,product_id,start,stop,tags)




