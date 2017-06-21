import luigi
import sqlite3
import pandas as pd

#from recommender import RecommendationEngine
#from pyspark import SparkContext

class UserTastePreprocess(luigi.Task):
	date_interval = luigi.DateIntervalParameter()

	def requires(self):
		return None

	def run(self):
		taste = pd.read_csv('./origin/train_triplets.txt', sep='\t', header=None, names=['user_id','song_id','play_count'])
		labels, levels = pd.factorize(taste['user_id'])
		taste['user_index'] = labels
		slabels, slevels = pd.factorize(taste['song_id'])
		taste['song_index'] = slabels
		# Export taste profile (encoded) to csv file
		# output data
		with self.output().open('w') as out_file:
			taste[['user_index','song_index','play_count','song_id']].to_csv(out_file, index=False)

	def output(self):
		return luigi.LocalTarget("./data/taste_profile_encoded_%s.csv" % self.date_interval)

class SongMetaPreprocess(luigi.Task):
	date_interval = luigi.DateIntervalParameter()

	def requires(self):
		return UserTastePreprocess(self.date_interval)

	def run(self):
		track_meta_path='./origin/track_metadata.db'
		conn = sqlite3.connect(track_meta_path)
		q = "SELECT song_id, title, artist_name FROM songs"
		res = conn.execute(q)
		echonest_meta = res.fetchall()
		song_meta = pd.DataFrame(echonest_meta, columns=['song_id','song_title','artist_name'])

		with self.input().open('r') as in_file:
			taste = pd.read_csv(in_file)
		song_encode = taste[['song_id','song_index']].drop_duplicates()

		song_encode_meta = pd.merge(song_encode, song_meta, how='left', on='song_id')

		with self.output().open('w') as out_file:
			song_encode_meta.drop('song_id', axis=1).to_csv(out_file, index=False, encoding='utf-8')

	def output(self):
		return luigi.LocalTarget("./data/song_encode_meta_%s.csv" % self.date_interval)
'''
class Subsample(luigi.Task):

	def require(self):
		return None

	def run(self):

	def output(self):

class TrainALS(luigi.Task):

	def require(self):
		return None

	def run(self):
	    sc = SparkContext()
	    #dataset_path = os.path.join('datasets', 'ml-latest')
	    recommendation_engine = RecommendationEngine(sc)

	def output(self):
		return luigi.LocalTarget("./data/recommender_model_%s" % self.date_interval)
'''