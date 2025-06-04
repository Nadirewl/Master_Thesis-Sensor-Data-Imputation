import pandas as pd
from data_insight import setup_duckdb
from duckdb import DuckDBPyConnection as DuckDB
from duckdb import DuckDBPyRelation as Relation


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
