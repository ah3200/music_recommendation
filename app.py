from flask import Blueprint
main = Blueprint('main', __name__)

import json
from recommender import RecommendationEngine

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from flask import Flask, request

@main.route("/<int:user_id>/ratings/top/<int:count>", methods=["GET"])
def top_reco(user_id, count):
	logger.debug("User %s TOP recommended songs requested", user_id)
	top_reco = recommendation_engine.get_top_n_reco(user_id, count)
	return json.dumps(top_reco)

@main.route("/<int:user_id>/ratings/<int:song_id>", methods=["GET"])
def song_ratings(user_id, song_id):
	logger.debug("User %s rating requested for song %s", user_id, song_id)
	ratings = recommendation_engine.get_ratings_for_song_ids(user_id, [song_id])
	return json.dumps(ratings)

@main.route("/<int:user_id>/ratings", methods=["POST"])
def add_ratings(user_id):
	ratings_list = request.form.keys()[0].strip().split("\n")
	ratings_list = map(lambda x: x.split(","), ratings_list)
	ratings = map(lambda x: (user_id, int(x[0]), float(x[1])), ratings_list)
	recommendation_engine.add_ratings(ratings)

	return json.dumps(ratings)

def create_app(spark_context, dataset_path):
	global recommendation_engine

	recommendation_engine = RecommendationEngine(spark_context, dataset_path)

	app = Flask(__name__)
	app.register_blueprint(main)
	return app

