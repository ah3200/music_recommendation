import os
import pyspark
import csv
import itertools

from pyspark import SparkContext
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.mllib.recommendation import ALS
import math

sc = SparkContext()

# Load Train file
taste_file = os.path.join('s3://music-recommendation','taste_profile.csv')
taste_raw_data = sc.textFile(taste_file)
taste_raw_data_header = taste_raw_data.take(1)[0]
# Remove first row as header, split each row into token
taste_data = taste_raw_data.filter(lambda line: line!=taste_raw_data_header)\
            .map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),int(tokens[1]),int(tokens[2]))).cache()

# Load Test file
test_file = os.path.join('s3://music-recommendation','subset_test_taste_profile.csv')
test_raw_data = sc.textFile(test_file)
test_raw_data_header = test_raw_data.take(1)[0]
# Remove first row as header, split each row into token
test_data = test_raw_data.filter(lambda line: line!=test_raw_data_header)\
            .map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),int(tokens[1]),int(tokens[2]))).cache()

print "Train"
print taste_data.count()
print "Test"
print test_data.count()
result_list = []

ranks = [10, 20, 40]
regularization_parameters  = [0.1, 1.0, 10.0]
iterations = [10, 20]
#ranks = [8]
#regularization_parameters  = [0.1]
#iterations = [10]
k = 500

for rank, regularization_param, numIter in itertools.product(ranks, regularization_parameters, iterations):
    model = ALS.trainImplicit(taste_data, rank, iterations=numIter, lambda_=regularization_param)
    print "Model built"
    userRecommended = model.recommendProductsForUsers(k)
    user_reco = userRecommended.map(lambda x: (x[0], [r.product for r in x[1]]))

    # Labels data
    user_songs = test_data.map(lambda x: (x[0], x[1])).groupByKey().mapValues(list)

    predictionAndLabels = user_reco.join(user_songs)
    test_predictionAndLabels = predictionAndLabels.map(lambda x: x[1])

    metrics = RankingMetrics(test_predictionAndLabels)
    result_list.append((rank, regularization_param, numIter, metrics.meanAveragePrecision, metrics.precisionAt(k), metrics.ndcgAt(k)))

    with open('./cross_val_result.csv', "a") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(result_list)
    result_list = []
#with open('s3://music-recommendation/cross_val_result.csv', "w") as output:
#    writer = csv.writer(output, lineterminator='\n')
#    writer.writerows(result_list)
#result = pd.DataFrame(result_list, columns=['rank','lambda','iteration','map','precision_k','ndcg_k'])
#result.to_csv('./cross_val_result.csv', index=False)
#result.to_csv('s3://music-recommendation/cross_val_result.csv', index=False)
# To run spark: 
# 1. unset PYSPARK_DRIVER_PYTHON
# 2. ~/spark-2.1.0-bin-hadoop2.7/bin/spark-submit --master local[2] --total-executor-cores 14 --executor-memory 4g server.py

# To run on AWS:
# 1. aws s3 cp s3://music-recommendation/recommender.py .
# 2. spark-submit recommender.py
