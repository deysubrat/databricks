from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window

'''spark = SparkSession.builder \
    .appName("CreditCarFraudDetection") \
    .getOrCreate()'''
    
customer_df = spark.read.option("header", "true").csv("/Workspace/Users/subratdey85@gmail.com/databricks/fraud_detection/data/customers.csv")
transaction_df = spark.read.option("header", "true").csv("/Workspace/Users/subratdey85@gmail.com/databricks/fraud_detection/data/transactions.csv")
merchant_df = spark.read.option("header", "true").csv("/Workspace/Users/subratdey85@gmail.com/databricks/fraud_detection/data/merchants.csv")
blacklist_df = spark.read.option("header", "true").csv("/Workspace/Users/subratdey85@gmail.com/databricks/fraud_detection/data/blacklist.csv")

transaction_df = transaction_df.dropDuplicates(["txn_id"])

transaction_df.write.mode("overwrite").parquet("/Workspace/Users/subratdey85@gmail.com/databricks/fraud_detection/bronze/transactions")

silver_df = transaction_df.join(customer_df, "customer_id", "left") \
    .join(merchant_df, "merchant_id", "left")

fraud_blacklist = silver_df.join(blacklist_df, 
                                 blacklist_df.account_id == silver_df.customer_id, 
                                 "inner")

high_value = silver_df.filter(silver_df.amount > 10000)

window_spec = Window.partitionBy("account_id").orderBy(col("txn_time"))

multiple_transactions = silver_df.withColumn("prev_time", lag("txn_time").over(window_spec))

known_devices = spark.createDataFrame([("ACC1001","D001"), ("ACC1002","D002")], ["account_id", "device_id"])
new_device_txn = silver_df.join(known_devices, ["account_id", "device_id"], "left_anti")

fraud_df = silver_df.withColumn(
    "fraud_score",
    when(col("amount") > 10000, 30)
    .otherwise(0)
)

fraud_df = fraud_df.withColumn(
    "fraud_score",
    when(col("amount") > 10000, 30)
    .otherwise(0)
    +
    when(col("account_id").isin(
        ["ACC1001", "ACC1005", "ACC1009"]
    ), 50)
    .otherwise(0)
)


fraud_df = fraud_df.withColumn(
    "risk_level",
    when(col("fraud_score") >= 70, "High")
    .when(col("fraud_score") >= 40, "Medium")
    .otherwise("Low")
)

fraud_df.write.mode("overwrite").parquet("/Workspace/Users/subratdey85@gmail.com/databricks/fraud_detection/gold/fraud_transactions")

print(fraud_df.filter(col("risk_level") == "High").count())

print(fraud_df.groupBy("city").agg(sum("amount").alias("fraud_amount")).show())
