import os 

PATH_TO_QUESTION = 'question/'
PATH_TO_TAG = 'tag/'
PATH_TO_DATA = 'data/'
class Utils():

	def load_data():
		question = {}
		for i in os.listdir(PATH_TO_QUESTION):
			question[i] = {}
			for filename in os.listdir(PATH_TO_QUESTION + i):
				question[i][filename] = []
				filein = open(PATH_TO_QUESTION + i + '/' + filename, 'r')
				for line in filein:
					question[i][filename].append(line)
		return question

	def load_tag():
		tag = {}
		for i in os.listdir(PATH_TO_TAG):
			tag[i] = {}
			for filename in os.listdir(PATH_TO_QUESTION + i):
				tag[i][filename] = []
				filein = open(PATH_TO_TAG + i + '/' + filename, 'r')
				for line in filein:
					tag[i][filename].append(line)
		return tag

	def load_nation():
		nation = []
		with open(PATH_TO_DATA + 'nation', 'r') as f:
			for line in f:
				line = line.replace('\n', '')
				nation.append(line)
		return nation

	def load_language():
		language = []
		with open(PATH_TO_DATA + 'language', 'r') as f:
			for line in f:
				line = line.replace('\n', '')
				language.append(line)
		return language

	def load_subject():
		subject = []
		with open(PATH_TO_DATA + 'subject', 'r') as f:
			for line in f:
				line = line.replace('\n', '')
				subject.append(line)
		return subject

	def load_participate():
		participate = []
		with open(PATH_TO_DATA + 'participate', 'r') as f:
			for line in f:
				line = line.replace('\n', '')
				participate.append(line)
		return participate

	def load_uncolleage():
		uncolleage = []
		with open(PATH_TO_DATA + 'uncolleage', 'r') as f:
			for line in f:
				line = line.replace('\n', '')
				uncolleage.append(line)
		return uncolleage

	def load_yesno():
		yesno = []
		with open(PATH_TO_DATA + 'yesno', 'r') as f:
			for line in f:
				line = line.replace('\n', '')
				yesno.append(line)
		return yesno

if __name__ == '__main__' :
	tag = Utils.load_tag()
	intend  = sorted(tag)
	direct = {}
	listtag = []
	for index in range(len(intend)):
		i = intend[index]
		direct = sorted(tag[i])
		for k in direct:
			tmp = tag[i][k]
			if (tmp not in listtag):
				listtag.append(tmp)
	print(listtag)
	print(len(Utils.load_nation()))

