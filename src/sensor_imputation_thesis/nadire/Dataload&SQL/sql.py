import data_insight
import pandas as pd
import os

#Get overlapped product id 
con = data_insight.setup_duckdb()
con.sql("SET enable_progress_bar=true")


ship_pid=con.sql("""
                 SELECT distinct productid, engine_type
                FROM shipinfo
                 WHERE engine_type LIKE '%ME%'
""")
 
print(ship_pid)

time_pid=con.sql("SELECT distinct pid FROM timeseries")

overlapped_id=con.sql("""
                  select s.productid, s.engine_type
                  from ship_pid s
                  join time_pid t ON s.productid=t.pid
                  """)
print(overlapped_id)

MEpidOverlapped_df =overlapped_id.df()
print(MEpidOverlapped_df)
MEpidOverlapped_df.to_parquet("/home/ec2-user/SageMaker/sensor-imputation-thesis/src/sensor_imputation_thesis/nadire/MEpidOverlapped_df.parquet", index=False)







#Get tags
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





