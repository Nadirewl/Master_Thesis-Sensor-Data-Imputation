import pandas as pd
from data_insight import setup_duckdb
from duckdb import DuckDBPyConnection as DuckDB
from duckdb import DuckDBPyRelation as Relation


if __name__ == "__main__":
    con = setup_duckdb()

    columns = con.sql("""
    SELECT column_name
    FROM (DESCRIBE timeseries)
    """).df().column_name.tolist()
    print('\n'.join(columns))