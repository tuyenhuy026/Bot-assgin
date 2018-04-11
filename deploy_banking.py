from model_banking import *
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

PATH_TO_RESPOND = 'respond/'
PATH_TO_QUESTION = 'question/'
PATH_TO_TAG = 'tag/'

grad = True

class PrivateInfo:
	question = Utils.load_data()['1']
	tag = Utils.load_tag()['1']
	intend = sorted(question)

	def __init__(self):
		self._numQuestion = len(self.question)
		self._questionCheck = np.zeros(self._numQuestion)
		self._endConversation = False
		self.counter = 0
		self._current = self.intend[self.counter]
		self._infomations = [None] * self._numQuestion

	def loadTag (self):
		return self.tag[self._current][0]

	def isEnd (self):
		if (self.counter == self._numQuestion):
			self._endConversation = True
		return self._endConversation

	def loadQuestion(self):
		ques = self.question[self._current]
		choice = random.randint(0, len(ques) - 1)
		return ques[choice]

	def gotInfo (self, ans, model, sess,
		transition_params, mapping_embedding, tag='ORG'):
		info = getInfomation(ans, tag, model, sess,
		 transition_params, mapping_embedding)
		print(info)
		if (info == ''):
			return False
		else :
			self._infomations[self.counter] = info
			self.counter += 1
			if (self.counter < self._numQuestion):
				self._current = self.intend[self.counter]
			return True

	def conservation (self, model, sess,
		transition_params, mapping_embedding):
		while not self.isEnd():
			Q = self.loadQuestion()
			T = self.loadTag()
			T = T.replace('\n', '')
			A = input(Q)
			info = ''
			if (self.counter == 6):
				tag = self.tag[self._current][1].replace('\n', '')
				info = getInfomation(A, tag, model, sess, 
					transition_params, mapping_embedding)
				if (info == ''):
					continue
			if (self.gotInfo(A, model, sess,
				transition_params, mapping_embedding, T)):
				self._infomations[self.counter - 1] += ' ' + info
				print (self._infomations[self.counter - 1])
				print('Infomation is updated')
			else :
				print ('Not updated')

class ParticipateInfo:
	question = Utils.load_data()['2']
	tag = Utils.load_tag()['2']
	intend = sorted(question)

	def __init__(self):
		self._numQuestion = len(self.question)
		self._questionCheck = np.zeros(self._numQuestion)
		self._endConversation = False
		self.counter = 0
		self._current = self.intend[self.counter]
		self._infomations = [None] * self._numQuestion

	def loadTag (self):
		return self.tag[self._current][0]

	def isEnd (self):
		if (self.counter == self._numQuestion):
			self._endConversation = True
		return self._endConversation

	def loadQuestion(self):
		ques = self.question[self._current]
		choice = random.randint(0, len(ques) - 1)
		return ques[choice]

	def gotInfo (self, ans, model, sess,
		transition_params, mapping_embedding, tag='ORG'):
		info = getInfomation(ans, tag, model, sess,
		 transition_params, mapping_embedding)
		print(info)
		if (info == ''):
			return False
		else :
			self._infomations[self.counter] = info
			self.counter += 1
			if (self.counter < self._numQuestion):
				self._current = self.intend[self.counter]
			return True

	def conservation (self, model, sess,
		transition_params, mapping_embedding):
		check = False
		while not self.isEnd():
			if (self._current == '2'):
				if (self._infomations[self.counter - 1] == 'sai'):
					grad = False
				else:
					grad = True
			if (self._current == '3'):
				if (self._infomations[self.counter - 1] != 'tự do'):
					self._infomations[self.counter] = 'NONE'
					self.counter += 1
					self._current = self.intend[self.counter]
					check = True
					continue
			print (check)
			if (check == True and self._current == '8'):
				break
			if (check == False and self._current == '7'):
				self.counter += 1
				self._current = self.intend[self.counter]
				continue
			Q = self.loadQuestion()
			T = self.loadTag()
			T = T.replace('\n', '')
			A = input(Q)
			if (self.gotInfo(A, model, sess,
				transition_params, mapping_embedding, T)):
				print('Infomation is updated')
			else :
				print ('Not updated')


class Graduate:
	question = Utils.load_data()['3']
	tag = Utils.load_tag()['3']
	intend = sorted(question)

	def __init__(self):
		self._numQuestion = len(self.question)
		self._questionCheck = np.zeros(self._numQuestion)
		self._endConversation = False
		self.counter = 0
		self._current = self.intend[self.counter]
		self._infomations = [None] * self._numQuestion

	def loadTag (self):
		return self.tag[self._current][0]

	def isEnd (self):
		if (self.counter == self._numQuestion):
			self._endConversation = True
		return self._endConversation

	def loadQuestion(self):
		ques = self.question[self._current]
		choice = random.randint(0, len(ques) - 1)
		return ques[choice]

	def gotInfo (self, ans, model, sess,
		transition_params, mapping_embedding, tag='ORG'):
		info = getInfomation(ans, tag, model, sess,
		 transition_params, mapping_embedding)
		print(info)
		if (info == ''):
			return False
		else :
			self._infomations[self.counter] = info
			self.counter += 1
			if (self.counter < self._numQuestion):
				self._current = self.intend[self.counter]
			return True

	def conservation (self, model, sess,
		transition_params, mapping_embedding):
		check = False
		while not self.isEnd():
			if (self._current == '2'):
				if (self._infomations[self.counter - 1] == 'sai'):
					self.counter += 1
					self._current = self.intend[self.counter]
					continue
			if (self._current == '4'):
				if (self._infomations[self.counter - 1] == 'sai'):
					break
			Q = self.loadQuestion()
			T = self.loadTag()
			T = T.replace('\n', '')
			A = input(Q)
			if (self.gotInfo(A, model, sess,
				transition_params, mapping_embedding, T)):
				print('Infomation is updated')
			else :
				print ('Not updated')	

class Contest:
	question = Utils.load_data()['4']
	tag = Utils.load_tag()['4']
	intend = sorted(question)

	def __init__(self):
		self._numQuestion = len(self.question)
		self._questionCheck = np.zeros(self._numQuestion)
		self._endConversation = False
		self.counter = 0
		self._current = self.intend[self.counter]
		self._infomations = [None] * self._numQuestion

	def loadTag (self):
		return self.tag[self._current][0]

	def isEnd (self):
		if (self.counter == self._numQuestion):
			self._endConversation = True
		return self._endConversation

	def loadQuestion(self):
		ques = self.question[self._current]
		choice = random.randint(0, len(ques) - 1)
		return ques[choice]

	def gotInfo (self, ans, model, sess,
		transition_params, mapping_embedding, tag='ORG'):
		info = getInfomation(ans, tag, model, sess,
		 transition_params, mapping_embedding)
		print(info)
		if (info == ''):
			return False
		else :
			self._infomations[self.counter] = info
			self.counter += 1
			if (self.counter < self._numQuestion):
				self._current = self.intend[self.counter]
			return True

	def conservation (self, model, sess,
		transition_params, mapping_embedding):
		check = False
		loop = 0
		tmp = []
		while not self.isEnd():
			if (self._current == '3'):
				if (self._infomations[self.counter - 1] == '2018'):
					self.counter += 1
					self._current = self.intend[self.counter]
					continue
			Q = self.loadQuestion()
			T = self.loadTag()
			T = T.replace('\n', '')
			A = input(Q)
			if (self.gotInfo(A, model, sess,
				transition_params, mapping_embedding, T)):
				print('Infomation is updated')
				if (self._current == '5'):
					loop = int(self._infomations[self.counter - 1])
				if (self.counter == 7 and loop > 0):
					loop -= 1
					tmp.append(self._infomations[self.counter - 1])
					self.counter = 4
					self._current = '5'
				if (loop == 0):
					self._infomations[self.counter - 1] = tmp 
			else :
				print ('Not updated')				


def main():
	date_text = ''
	sess = tf.Session()

	pretrained_model_folder = MODELS_FOLDER + '/vi.vec_100d_89_f1_macro'
	# Load mapping and embedding
	with open(pretrained_model_folder + MAPPING_EMBEDDING_FILE, 'rb') as file:
		mapping_embedding = pickle.load(file)
	# Init and restore model
	model = EntityLSTM(mapping_embedding.embedding_tensor, 32)
	# Transition parameters for viterbi decode
	transition_params = model.restore_pretrained_model(sess, pretrained_model_folder)


	demo = Contest()
	demo.conservation(model, sess, transition_params, mapping_embedding)


if __name__ == '__main__':
	main()