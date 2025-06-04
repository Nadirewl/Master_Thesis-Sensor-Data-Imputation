import pandas as pd
from data_insight import setup_duckdb
from duckdb import DuckDBPyConnection as DuckDB
from duckdb import DuckDBPyRelation as Relation


def load_engine_data(
    con: DuckDB, product_id: str, start: pd.Timestamp, stop: pd.Timestamp, tags: list[str]
) -> Relation:
    con = setup_duckdb()

    con.sql(f"""
    SELECT {tuple(tags)}
    FROM timeseries
    WHERE
        time BETWEEN '{start}' AND '{stop}'
        AND pid = '{product_id}'
    """)


if __name__ == "__main__":
    con = setup_duckdb()

    con.sql("""
    SELECT DISTINCT pid
    FROM timeseries
    """)

    con.sql("""
    SELECT fr_eng
    FROM timeseries
    LIMIT 1
    """)

    start, stop = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-01")
    tags = [
        "time",
        "fr_eng",
        "an_phase_setpoint",
        "bo_gov_mode_index",
        "pd_cw_liner",
        "pr_exh_turb_out",
    ]
    product_id = "4379247-1"

    data = load_engine_data(con, product_id, start, stop, tags)
    data.df()


#Loading Dataframe

start, stop = pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-01")
tags = [
        "time",
        "fr_eng",
        "an_phase_setpoint",
        "bo_gov_mode_index",
        "pd_cw_liner",
        "pr_exh_turb_out__0",

    ]
product_id = "4379247-1"

cache = Path('/tmp/data.parquet')
if cache.exists():
    df = pd.read_parquet(cache)
else:
    con = setup_duckdb()
    df = load.load_engine_data(con,product_id,start,stop,tags).df()
    df.to_parquet(cache)
    
print(df)

