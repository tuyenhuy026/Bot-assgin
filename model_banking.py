import datetime
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
import re
from utils import Utils

def valid_code(str_):
	code = re.findall(r'[A-Z]\d\d', str_)
	if (len(code)):
		return code[0]
	return ''

def valid_email(str_):
	email = re.findall(r'\d*[a-z]+\d*@[a-z]+.[a-z]+', str_)
	if (len(email)):
		return email[0]
	return ''

def valid_number(str_):
	num = re.findall(r'\d+', str_)
	if (len(num)):
		return num[0]
	return ''

def valid_nation(str_):
	str_ = str_.lower()
	nation = Utils.load_nation()
	for i in nation:
		if i in str_:
			return i
	return ''

def valid_language(str_):
	str_ = str_.lower()
	language = Utils.load_language()
	for i in language:
		if i in str_:
			return i
	return ''

def valid_subject(str_):
	str_ = str_.lower()
	subject = Utils.load_subject()
	out = []
	for i in subject:
		if i in str_:
			out.append(i)
	if (len(out) > 0):
		return out
	return ''

def get_subject_score(str_):
	str_ = str_.lower()
	subject = valid_subject(str_)
	str_ = str_.split()
	out = []
	for word in str_:
		for i in subject:
			if i in word:
				out.append(word)
	return out

def valid_participate(str_):
	str_ = str_.lower()
	participate = Utils.load_participate()
	for i in participate:
		if i in str_:
			return i
	return ''

def valid_uncolleage(str_):
	str_ = str_.lower()
	uncolleage = Utils.load_uncolleage()
	for i in uncolleage:
		if i in str_:
			return i
	return ''


def isDate(date_text, isTime = False):
	if (isTime):
		try:
			date = datetime.datetime.strptime(date_text, '%H-%d-%m-%Y')
		except ValueError:
			return ''
	else:
		try:
			date = datetime.datetime.strptime(date_text, '%d-%m-%Y')
			return date_text
		except ValueError:
			return ''
	return date_text

def valid_date(message, isTime = False):
	filtered = message.split()
	output = ''
	for date_text in filtered:
		print (isDate(date_text, isTime))
		print (isDate(date_text, isTime) == output)
		if (isDate(date_text, isTime) != output):
			print(date_text)
			output = date_text
	if (isTime):
		return 'No change'
	return output

def valid_nameEntity(message, model, tag, sess, transition_params, mapping_embedding):

	print(tag)
	result = model.predict(sess, transition_params, mapping_embedding, message)
	nameEntity = ''
	for pair in result:
		if (pair[1] != tag):
			continue
		for i in range(len(pair[0])):
			nameEntity = nameEntity + pair[0][i] + ' '
		message = message.replace(nameEntity, tag)
	return nameEntity

def valid_yesno(str_):
	str_ = str_.lower()
	yesno = Utils.load_yesno()
	for i in yesno:
		if i in str_:
			return i
	return ''

def valid_pioritize(str_):
	piority = re.findall(r'0\d', str_)
	if(len(piority)):
		return piority[0]
	return ''

def getInfomation(message, tag, model, sess, transition_params, mapping_embedding):

	if (tag == 'PER'):
		return valid_nameEntity(message, model, tag, sess, transition_params, mapping_embedding)
	if (tag == 'NUM'):
		return valid_number(message)
	if (tag == 'LOC'):
		return valid_nameEntity(message, model, tag, sess, transition_params, mapping_embedding)
	if (tag == 'DATE'):
		return valid_date(message)
	if (tag == 'PER'):
		return valid_nameEntity(message, model, tag, sess, transition_params, mapping_embedding)
	if (tag == 'ORG'):
		return valid_nameEntity(message, model, tag, sess, transition_params, mapping_embedding)
	if (tag == 'EMAIL'):
		return valid_email(message)
	if (tag == 'NATION'):
		return valid_nation(message)
	if (tag == 'PARTICIPATE'):
		return valid_participate(message)
	if (tag == 'LANGUAGE'):
		return valid_language(message)
	if (tag == 'SUBJECT'):
		return valid_subject(message)
	if (tag == 'CERTIFICATE'):
		return message
	if (tag == 'UNCOLLEAGE'):
		return valid_uncolleage(message)
	if (tag == 'CODE'):
		return valid_code(message)
	if (tag == 'YESNO'):
		return valid_yesno(message)
	if (tag == 'SUBJECTSCORE'):
		return get_subject_score(message)
	if (tag == 'PIORITY'):
		return valid_pioritize(message)

if __name__ == '__main__' :
	message = input()
	print(valid_nation(message))
	sess = tf.Session()

	pretrained_model_folder = MODELS_FOLDER + '/vi.vec_100d_89_f1_macro'
	# Load mapping and embedding
	with open(pretrained_model_folder + MAPPING_EMBEDDING_FILE, 'rb') as file:
		mapping_embedding = pickle.load(file)
	# Init and restore model
	model = EntityLSTM(mapping_embedding.embedding_tensor, 32)
	# Transition parameters for viterbi decode
	transition_params = model.restore_pretrained_model(sess, pretrained_model_folder)
	print(getInfomation(message, 'PER', model, sess, transition_params, mapping_embedding))
