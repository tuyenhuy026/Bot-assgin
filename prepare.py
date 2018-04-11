import json
import os
import io 
import numpy as np

data = {}

try:
    to_unicode = unicode
except NameError:
    to_unicode = str

while (True):
	check = input('Do you want to continue ? y/n \n')

	if (check == 'y'):
		infomation = input('Input below infomation what you want to add \n')
		data[infomation] = []
		if (os.path.isfile(infomation)):
			f = open(infomation, 'r')
			for question in f:
				data[infomation].append(question)
			f.close()
		else:
			for i in range(5):
				question = input('Input question \n')
				data[infomation].append(question)
	else :
		print('Write file out')
		print(data)
		np_data = np.array(data)
		np.save('data', np_data)
		print('Complete')
		break