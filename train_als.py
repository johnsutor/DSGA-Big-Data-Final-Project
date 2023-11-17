#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Main code behind the final project for Group 69
Usage:
    $ spark-submit final_project.py
"""

import os
import time
import datetime
from itertools import product

import numpy as np
import pandas as pd
import pprint

from pyspark.sql import SparkSession, Window
import pyspark.sql.functions as F
from pyspark.sql.functions import (
    max,
    sum,
    col,
    collect_list,
    when,
    explode,
)
from pyspark.ml.evaluation import RankingEvaluator
from pyspark.ml.recommendation import ALS

from scipy.stats import loguniform, randint
from itertools import product


def main(spark, **kwargs):
    """Main routine for training the recommendation system
    Parameters
    ----------
    spark : SparkSession object

    Keyword Arguments
    -----------------
    small_data: Boolean representing whether or not to use small data
    """
    DATA_DIR = "hdfs:/user/bm106_nyu_edu/1004-project-2023/"
    # interactions_train = spark.read.parquet(os.path.join(DATA_DIR, "interactions_split_train.pqt")
    # interactions_val = spark.read.parquet(os.path.join(DATA_DIR, "interactions_split_val.pqt")

    def gen_utiltiy_matrix(interactions, tracks):
        interactions.createOrReplaceTempView("interactions")
        tracks.createOrReplaceTempView("tracks")

        listens_per_user_track = spark.sql(
            """
            SELECT user_id,universal_id,sum(num_listens) as num_listens
            FROM interactions
            LEFT JOIN tracks
            ON tracks.recording_msid=interactions.recording_msid
            GROUP BY user_id,universal_id
            """
        )

        listens_per_user = (
            listens_per_user_track.select(
                listens_per_user_track.user_id, listens_per_user_track.num_listens
            )
            .groupBy("user_id")
            .agg(sum(listens_per_user_track.num_listens).alias("total_listens"))
        )

        listens_per_user = listens_per_user.withColumn(
            "use_for_fit",
            when(listens_per_user.total_listens >= 500, True).otherwise(False),
        )

        normed_listens_per_user_track = listens_per_user_track.join(
            listens_per_user, how="left", on="user_id"
        )
        normed_listens_per_user_track = (
            normed_listens_per_user_track.withColumn(
                "prop_listens", col("num_listens") / col("total_listens")
            )
            .select(["user_id", "universal_id", "prop_listens", "use_for_fit"])
            .orderBy(col("user_id").asc(), col("prop_listens").desc())
        )

        return normed_listens_per_user_track

    interactions_train = spark.read.parquet("interactions_split_train.parquet")
    interactions_val = spark.read.parquet("interactions_split_val.parquet")
    interactions_test = spark.read.parquet(
        "/scratch/work/courses/DSGA1004-2021/listenbrainz/interactions_test.parquet"
    )

    tracks_train = spark.read.parquet("tracks_train.parquet")
    tracks_test = spark.read.parquet("tracks_test.parquet")

    utility_mat_train = gen_utiltiy_matrix(interactions_train, tracks_train)
    utility_mat_val = gen_utiltiy_matrix(interactions_val, tracks_train)
    utility_mat_test = gen_utiltiy_matrix(interactions_test, tracks_test)

    utility_mat_train = utility_mat_train.filter(utility_mat_train.use_for_fit)

    def calc_performance_metrics_als(predicted, actual):
        actual_compressed = actual.groupBy("user_id").agg(
            collect_list(col("universal_id").astype("double")).alias("universal_id"),
            collect_list(col("prop_listens").astype("double")).alias("prop_listens"),
        )

        predicted_compressed = predicted.withColumn(
            "recommendations", explode(col("recommendations"))
        ).select("user_id", "recommendations.universal_id", "recommendations.rating")

        predicted_compressed = (
            predicted_compressed.withColumn(
                "rn",
                F.row_number().over(
                    Window.partitionBy("user_id").orderBy(F.col("rating").desc())
                ),
            )
            .groupBy("user_id")
            .agg(
                F.collect_list(F.col("universal_id"))
                .astype("array<double>")
                .alias("predicted_universal_id")
            )
        )

        results = actual_compressed.join(predicted_compressed, how="left", on="user_id")

        mapAtK = RankingEvaluator(
            predictionCol="predicted_universal_id",
            labelCol="universal_id",
            metricName="meanAveragePrecisionAtK",
            k=100,
        )

        ndcgAtK = RankingEvaluator(
            predictionCol="predicted_universal_id",
            labelCol="universal_id",
            metricName="ndcgAtK",
            k=100,
        )
        return (mapAtK.evaluate(results), ndcgAtK.evaluate(results))

    # Handle different parameters
    # regParams = [1] # [pow(10., i) for i in range(-5, 5)]
    # ranks = [i for i in range(10, 6000)]
    # alphas = [1] # [pow(10., i) for i in range(-5, 5)]

    DEBUG = True
    SEED = 69
    TRIALS = 100

    # Utilize random search to find optimal hyperparameters
    rank_min = 1
    rank_max = 10

    # alpha_min = 1e-5
    # alpha_max = 1e5

    reg_param_min = 1e-5
    reg_param_max = 1e5

    max_iter_min = 10
    max_iter_max = 15

    training_results = []
    start = time.perf_counter()

    for t in range(TRIALS):
        regParam = loguniform(reg_param_min, reg_param_max).rvs()
        rank = randint(rank_min, rank_max).rvs()
        # alpha = loguniform(alpha_min, alpha_max).rvs()
        maxIter = randint(max_iter_min, max_iter_max).rvs()

        als = ALS(
            maxIter=maxIter,
            implicitPrefs=True,
            nonnegative=True,
            regParam=regParam,
            rank=rank,
            # alpha=alpha,
            seed=SEED,
            userCol="user_id",
            itemCol="universal_id",
            ratingCol="prop_listens",
            coldStartStrategy="drop",
        )

        # Calculate MAP
        model = als.fit(utility_mat_train)

        predictions_train = model.recommendForUserSubset(
            utility_mat_train.select("user_id").distinct(), 100
        )
        predictions_val = model.recommendForUserSubset(
            utility_mat_val.select("user_id").distinct(), 100
        )

        map_train, ndcg_train = calc_performance_metrics_als(
            predictions_train, utility_mat_train
        )
        map_val, ndcg_val = calc_performance_metrics_als(
            predictions_val, utility_mat_val
        )

        elapsed = datetime.timedelta(seconds=time.perf_counter() - start)

        training_results.append(
            {
                "maxIter": maxIter,
                "elapsed": str(elapsed),
                "regParam": regParam,
                "rank": rank,
                # "alpha": alpha,
                "map_train": map_train,
                "map_val": map_val,
                "ndcg_train": ndcg_train,
                "ndcg_val": ndcg_val,
                "trial": t + 1,
            }
        )

        pprint.pprint(training_results[-1], width=1)

    pd.DataFrame(training_results).to_csv("als_results.csv")


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName("part1").getOrCreate()

    # Get user userID from the command line
    # We need this to access the user's folder in HDFS
    # userID = os.environ['USER']

    # Call our main routine
    main(spark, small_data=True)
