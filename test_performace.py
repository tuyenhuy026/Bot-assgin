from model import EntityLSTM
import pickle
import tensorflow as tf
from constants import *
import time
from dataset import Dataset, MappingAndEmbedding
import codecs
import random

def read_doc(filepath):
    text = ''
    with codecs.open(filepath, 'r', 'utf-8') as doc:
        for line in doc:
            text += line
    return text

def main():
    sess = tf.Session()
    pretrained_model_folder = MODELS_FOLDER + '/vi.vec_100d_89_f1_macro'
    
    #load mapping and embedding
    with open(pretrained_model_folder + MAPPING_EMBEDDING_FILE, 'rb') as file:
        mapping_embedding = pickle.load(file)
    
    #init and restore model
    model = EntityLSTM(mapping_embedding.embedding_tensor, 32)
    #transition parameters for viterbi decode
    transition_params = model.restore_pretrained_model(sess, pretrained_model_folder)
    
    #deploy loop
    test_folder = '../performance_test_doc/'
    sum = 0
    indices = [1,2,3,4,5,6,7,8,9,10]
    for i in range(100):
        random.shuffle(indices)
        for index in indices:
            doc = test_folder + str(index)
            text = read_doc(doc)
            start = time.time()
            model.predict(sess, transition_params, mapping_embedding, text)
            end = time.time()
            print('Iter {}, index {}, total time: {}'.format(i, index, end-start))
            sum += end-start
    
    print('Average: {}'.format(sum/1000))
    sess.close()

if __name__ == "__main__":
    main()