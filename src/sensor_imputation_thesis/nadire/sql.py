import data_insight
import pandas as pd
import os
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








engine_type=con.sql("SELECT distinct engine_type FROM shipinfo")
print(engine_type)

engine_type_df = engine_type.df()
# Display the full result
print(engine_type_df)
# Save to CSV
engine_type_df.to_parquet("/home/ec2-user/SageMaker/sensor-imputation-thesis/src/sensor_imputation_thesis/nadire/engine_types.parquet", index=False)

#con.sql("SELECT distinct ship_name FROM shipinfo")


pid = "237848bc2ed636c1ac85baea4755344c"
result = con.sql(f"SELECT engine_type FROM shipinfo WHERE productId = '{pid}'").df()
print("Engine Types:")
print(result)

engine_uuid = con.sql(f"SELECT uuid FROM site_infos WHERE product_id = '{pid}'").df()
print("\nEngine UUIDs:")
print(engine_uuid)











