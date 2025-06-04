import pandas as pd
from data_insight import setup_duckdb
from duckdb import DuckDBPyConnection as DuckDB
from duckdb import DuckDBPyRelation as Relation

if __name__ == "__main__":
    con = setup_duckdb()

    # Retrieve column names from the timeseries table
    column_names = con.sql("""
    SELECT column_name
    FROM information_schema.columns
    WHERE table_name = 'timeseries'
    """).df()

    print("Column Names in 'timeseries':")
    print(column_names)
