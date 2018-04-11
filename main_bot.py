from flask import Flask, render_template, request
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

from deploy_banking import cilent
import model_banking
# import faiss
import fastText as ft 
import numpy as np 
import random
from model import EntityLSTM
import pickle
import tensorflow as tf
from constants import *
from dataset import Dataset, MappingAndEmbedding
import os
from utils import Utils

sess = tf.Session()

pretrained_model_folder = MODELS_FOLDER + '/vi.vec_100d_89_f1_macro'
# Load mapping and embedding
with open(pretrained_model_folder + MAPPING_EMBEDDING_FILE, 'rb') as file:
	mapping_embedding = pickle.load(file)
# Init and restore model
model = EntityLSTM(mapping_embedding.embedding_tensor, 32)
# Transition parameters for viterbi decode
transition_params = model.restore_pretrained_model(sess, pretrained_model_folder)

demo = cilent()


app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
	ans = request.args.get('msg')
	if demo.start:
		question = demo.loadQuestion()
		return question
	if not demo.isEnd():
		tag = demo.loadIntent()
		if (demo.gotInfo(ans, model, sess, transition_params, mapping_embedding, tag)):
			print('Updated infomation')
			demo.display()
		if (demo.isEnd()):
			return demo.validInfo()
		question = demo.loadQuestion()
		return question
	else:
		return demo.validInfo()

if __name__ == "__main__":
    app.run()