from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType

# Create a Spark session
spark = SparkSession.builder.appName("Advanced Flight Data Analysis").getOrCreate()

# Load datasets
flights_df = spark.read.csv("flights.csv", header=True, inferSchema=True)
airports_df = spark.read.csv("airports.csv", header=True, inferSchema=True)
carriers_df = spark.read.csv("carriers.csv", header=True, inferSchema=True)

# Define output paths
output_dir = "output/"
task1_output = output_dir + "task1_largest_discrepancy.csv"
task2_output = output_dir + "task2_consistent_airlines.csv"
task3_output = output_dir + "task3_canceled_routes.csv"
task4_output = output_dir + "task4_carrier_performance_time_of_day.csv"

# ------------------------
# Task 1: Flights with the Largest Discrepancy Between Scheduled and Actual Travel Time
# ------------------------
def task1_largest_discrepancy(flights_df, carriers_df):
    # Calculate scheduled and actual travel times in minutes
    flights_df = flights_df.withColumn(
        "scheduled_travel_time",
        (F.unix_timestamp("ScheduledArrival") - F.unix_timestamp("ScheduledDeparture")) / 60
    ).withColumn(
        "actual_travel_time",
        (F.unix_timestamp("ActualArrival") - F.unix_timestamp("ActualDeparture")) / 60
    ).withColumn(
        "discrepancy",
        F.abs(F.col("scheduled_travel_time") - F.col("actual_travel_time"))
    )

    # Join with carriers to get carrier names
    result_df = flights_df.join(
        carriers_df,
        flights_df["CarrierCode"] == carriers_df["CarrierCode"]
    ).select(
        "FlightNum", 
        carriers_df["CarrierName"],
        "Origin",
        "Destination",
        "scheduled_travel_time",
        "actual_travel_time",
        "discrepancy",
        flights_df["CarrierCode"]
    )

    # Define a window to rank discrepancies within each carrier
    window_spec = Window.partitionBy("CarrierCode").orderBy(F.desc("discrepancy"))

    # Apply the window function to get the largest discrepancy per carrier
    ranked_df = result_df.withColumn("rank", F.row_number().over(window_spec))

    # Filter to keep only the top-ranked discrepancies
    largest_discrepancy = ranked_df.filter(ranked_df["rank"] == 1).select(
        "FlightNum",
        "CarrierName",
        "Origin",
        "Destination",
        "scheduled_travel_time",
        "actual_travel_time",
        "discrepancy",
        "CarrierCode"
    )

    # Write the result to a CSV file
    largest_discrepancy.write.csv(task1_output, header=True)
    print(f"Task 1 output written to {task1_output}")
task1_largest_discrepancy(flights_df, carriers_df)

# ------------------------
# Task 2: Most Consistently On-Time Airlines Using Standard Deviation
# ------------------------
def task2_consistent_airlines(flights_df, carriers_df):
    # Calculate departure delay in minutes
    flights_df = flights_df.withColumn(
        "departure_delay",
        (F.unix_timestamp("ActualDeparture") - F.unix_timestamp("ScheduledDeparture")) / 60  # in minutes
    )
    
    # Aggregate data to get standard deviation and count of flights per carrier
    delay_stats_df = flights_df.groupBy("CarrierCode").agg(
        F.count("departure_delay").alias("num_flights"),
        F.stddev("departure_delay").alias("stddev_departure_delay")
    ).filter("num_flights > 100")  # Only include carriers with more than 100 flights
    
    # Join with carriers to get carrier names
    delay_stats_df = delay_stats_df.join(
        carriers_df, delay_stats_df["CarrierCode"] == carriers_df["CarrierCode"]
    ).select(
        carriers_df["CarrierName"],
        "num_flights",
        "stddev_departure_delay"
    ).orderBy("stddev_departure_delay")  # Order by consistency (smallest standard deviation)

    # Write the result to a CSV file
    delay_stats_df.write.csv(task2_output, header=True)
    print(f"Task 2 output written to {task2_output}")

# Call the function to run Task 2
task2_consistent_airlines(flights_df, carriers_df)

# ------------------------
# Task 3: Origin-Destination Pairs with the Highest Percentage of Canceled Flights
# ------------------------

def task3_canceled_routes(flights_df, airports_df):
    # Identify canceled flights
    flights_df = flights_df.withColumn(
        "is_canceled", F.when(F.col("ActualDeparture").isNull(), 1).otherwise(0)
    )

    # Calculate total flights and cancellation rate for each origin-destination pair
    route_cancellation_df = flights_df.groupBy("Origin", "Destination").agg(
        F.count("FlightNum").alias("total_flights"),
        F.sum("is_canceled").alias("canceled_flights")
    ).withColumn(
        "cancellation_rate", 
        (F.col("canceled_flights").cast(DoubleType()) / F.col("total_flights")) * 100
    )

    # Join with airports_df to get names and cities for both origin and destination
    route_with_airports_df = route_cancellation_df \
        .join(airports_df.withColumnRenamed("AirportCode", "Origin")
              .withColumnRenamed("AirportName", "OriginAirportName")
              .withColumnRenamed("City", "OriginCity"), "Origin") \
        .join(airports_df.withColumnRenamed("AirportCode", "Destination")
              .withColumnRenamed("AirportName", "DestinationAirportName")
              .withColumnRenamed("City", "DestinationCity"), "Destination") \
        .select(
            "Origin",
            "OriginAirportName",
            "OriginCity",
            "Destination",
            "DestinationAirportName",
            "DestinationCity",
            "cancellation_rate"
        ).orderBy(F.desc("cancellation_rate"))  # Order by highest cancellation rate

    # Write the result to a CSV file
    route_with_airports_df.write.csv(task3_output, header=True)
    print(f"Task 3 output written to {task3_output}")

# Call the function to run Task 3
task3_canceled_routes(flights_df, airports_df)


# ------------------------
# Task 4: Carrier Performance Based on Time of Day
# ------------------------
def task4_carrier_performance_time_of_day(flights_df, carriers_df):
    # Convert ScheduledDeparture to hour to categorize time of day
    flights_df = flights_df.withColumn(
        "ScheduledHour", F.hour(F.col("ScheduledDeparture"))
    )

    # Define time of day based on hour
    flights_df = flights_df.withColumn(
        "time_of_day",
        F.when((F.col("ScheduledHour") >= 6) & (F.col("ScheduledHour") < 12), "Morning")
         .when((F.col("ScheduledHour") >= 12) & (F.col("ScheduledHour") < 18), "Afternoon")
         .when((F.col("ScheduledHour") >= 18) & (F.col("ScheduledHour") < 24), "Evening")
         .otherwise("Night")
    )

    # Calculate departure delay in minutes
    flights_df = flights_df.withColumn(
        "departure_delay",
        (F.unix_timestamp("ActualDeparture") - F.unix_timestamp("ScheduledDeparture")) / 60
    )

    # Calculate the average delay by carrier and time of day
    avg_delay_df = flights_df.groupBy("CarrierCode", "time_of_day").agg(
        F.avg("departure_delay").alias("avg_departure_delay")
    )

    # Rank carriers within each time period based on average delay
    window_spec = Window.partitionBy("time_of_day").orderBy("avg_departure_delay")
    ranked_carriers_df = avg_delay_df.withColumn(
        "rank", F.row_number().over(window_spec)
    )

    # Join with carriers to get the carrier name
    result_df = ranked_carriers_df.join(
        carriers_df, ranked_carriers_df.CarrierCode == carriers_df.CarrierCode
    ).select(
        "CarrierName", "time_of_day", "avg_departure_delay", "rank"
    ).orderBy("time_of_day", "rank")

    # Write the result to a CSV file
    result_df.write.csv(task4_output, header=True)
    print(f"Task 4 output written to {task4_output}")

task4_carrier_performance_time_of_day(flights_df, carriers_df)

# Stop the Spark session
spark.stop()