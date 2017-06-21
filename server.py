import time, sys, cherrypy, os
from paste.translogger import TransLogger 
from app import create_app
from pyspark import SparkContext, SparkConf

def init_spark_context():
	conf = SparkConf().setAppName("music_recommendation_server")
	sc = SparkContext(conf=conf, pyFiles=['recommender.py','app.py'])

	return sc

def run_server(app):

	# Enable WSGI access logging via Paste
	app_logged = TransLogger(app)

	# Mount the WSGI callable object (app) on the root directory
	cherrypy.tree.graft(app_logged, '/')

	# Config
	cherrypy.config.update({
		'engine.autoreload.on': True,
		'log.screen': True,
		'server.socket_port': 5432,
		'server.socket_host': '0.0.0.0'
		})

	# Start CherryPy WSGI web server
	cherrypy.engine.start()
	cherrypy.engine.block()

if __name__ == "__main__":
	sc = init_spark_context()
	dataset_path = os.path.join('.')
	app = create_app(sc, dataset_path)

	# start web server
	run_server(app)

	#~/spark-1.3.1-bin-hadoop2.6/bin/spark-submit --master spark://169.254.206.2:7077 --total-executor-cores 14 --executor-memory 6g server.py

