import os
from pyspark.mllib.recommendation import ALS
from pyspark.mllib.recommendation import MatrixFactorizationModel

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_counts_and_averages(ID_and_playcount_tuple):
    """Given a tuple (songID, playcount_iterable) 
    returns (songID, (playcount, playcount_avg))
    """
    nplays = len(ID_and_playcount_tuple[1])
    return ID_and_playcount_tuple[0], (nplays, float(sum(x for x in ID_and_playcount_tuple[1]))/nplays)

class RecommendationEngine:
    def __count_and_average_ratings(self):
        # Updates the play counts from the current data self.ratings_RDD
        logger.info("Counting song play counts...")
        songID_with_playcount_RDD = self.taste_RDD.map(lambda x: (x[1], x[2])).groupByKey()
        songID_with_avg_playcount_RDD = songID_with_playcount_RDD.map(get_counts_and_averages)
        self.song_playcounts_RDD = songID_with_avg_playcount_RDD.map(lambda x: (x[0], x[1][0]))

    def __train_model(self):
        """Train the ALS model with the current dataset
        """
        logger.info("Training the ALS model...")
        self.model = ALS.trainImplicit(self.taste_RDD, self.rank, seed=self.seed,
                               iterations=self.iterations, lambda_=self.regularization_parameter)
        logger.info("ALS model built!")

    def __persist_model(self):
        self.model.save(self.sc, "model1")
        logger.info("Saved ALS model !!")

    ####
    # Predicit rating (or confidence of number of play counts)
    def __predict_ratings(self, user_and_song_RDD):
        """Gets predictions for a given (userID, songID) formatted RDD
        Returns: an RDD with format (movieTitle, movieRating, numRatings)
        """
        user_predicted_RDD = self.model.predictAll(user_and_song_RDD)
        predicted_for_user = user_predicted_RDD.map(lambda x: (x.product, x.rating))
        user_predicted_title_and_count_RDD = \
            predicted_for_user.join(self.song_RDD).join(self.song_playcounts_RDD)
        user_predicted_title_and_count_RDD = \
            user_predicted_title_and_count_RDD.map(lambda r: (r[1][0][1], r[1][0][0], r[1][1]))
        
        return user_predicted_title_and_count_RDD

    def add_ratings(self, ratings):
        """Add additional movie ratings in the format (user_id, movie_id, rating)
        """
        # Convert data into an RDD
        new_playcounts_RDD = self.sc.parallelize(playcounts)
        # Add new data to the existing ones
        self.taste_RDD = self.taste_RDD.union(new_ratings_RDD)
        # Re-compute song play count
        self.__count_and_average_ratings()
        # Re-train the ALS model with the new data (users' playcounts)
        self.__train_model()
        
        return ratings

#    def get_ratings_for_movie_ids(self, user_id, movie_ids):
#        """Given a user_id and a list of movie_ids, predict ratings for them 
#        """
#        requested_movies_RDD = self.sc.parallelize(movie_ids).map(lambda x: (user_id, x))
#        # Get predicted ratings
#        ratings = self.__predict_ratings(requested_movies_RDD).collect()
#
#        return ratings
    
    def get_top_n_reco(self, user_id, n_songs):
        """Recommends up to n_songs to user_id
        """
        # Get pairs of (userID, movieID) for user_id unrated movies
        user_unplayed_songs_RDD = self.taste_RDD.filter(lambda rating: not rating[0] == user_id)\
                                                 .map(lambda x: (user_id, x[1])).distinct()
        # Get predicted ratings
        top_n_reco = self.__predict_ratings(user_unplayed_songs_RDD).filter(lambda r: r[2]>=25).takeOrdered(n_songs, key=lambda x: -x[1])

        return top_n_reco

    def __init__(self, sc, dataset_path='.'):
        """Init the recommendation engine given a Spark context and a dataset path
        """
 
        logger.info("Starting up the Recommendation Engine: ")
 
        self.sc = sc
 
        # Load user's taste (playcount) data
        #taste_file = os.path.join(dataset_path,'data','subset_taste_profile.csv')
        taste_file = os.path.join(dataset_path,'data','subset_taste_profile.csv')
        taste_raw_data = sc.textFile(taste_file)
        taste_raw_data_header = taste_raw_data.take(1)[0]
        self.taste_RDD = taste_raw_data.filter(lambda line: line!=taste_raw_data_header)\
                    .map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),int(tokens[1]),int(tokens[2]))).cache()
        # Load song file
        song_file = os.path.join('.','data','song_encode_meta.csv')
        song_raw_data = sc.textFile(song_file)
        song_raw_data_header = song_raw_data.take(1)[0]
        self.song_RDD = song_raw_data.filter(lambda line: line!=song_raw_data_header)\
                   .map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),tokens[1],tokens[2])).cache()
        
        # Pre-calculate song count
        self.__count_and_average_ratings()

        # Load model
        self.model = MatrixFactorizationModel.load(sc, "model_full")
        logger.info("Recommendation Engine Model is Loaded Successfully ... ")

        # Train the model
        #self.rank = 40
        #self.seed = 5L
        #self.iterations = 10
        #self.regularization_parameter = 0.1
        #self.__train_model()
        #self.__persist_model()

#if __name__ == "__main__":
#    # Init spark context and load libraries
#    from pyspark import SparkContext
#    sc = SparkContext()
#    #dataset_path = os.path.join('datasets', 'ml-latest')
#    recommendation_engine = RecommendationEngine(sc)

# To run spark: 
# 1. unset PYSPARK_DRIVER_PYTHON
# 2. ~/spark-2.1.0-bin-hadoop2.7/bin/spark-submit --master local[2] --total-executor-cores 14 --executor-memory 4g server.py