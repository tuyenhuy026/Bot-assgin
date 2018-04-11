from dataset import Dataset, MappingAndEmbedding
from model import EntityLSTM
from constants import *
import data_util
import numpy as np
import os
import pickle
import tensorflow as tf

def main():
    token2vec = data_util.load_token_embeddings()
    token_sequences, label_sequences = data_util.load_raw_data()
    mapping_embedding = MappingAndEmbedding(token2vec)

    train_set = Dataset('train', token_sequences[TRAIN], label_sequences[TRAIN], mapping_embedding)
    dev_set = Dataset('develop', token_sequences[DEV], label_sequences[DEV], mapping_embedding)
    test_set = Dataset('test', token_sequences[TEST], label_sequences[TEST], mapping_embedding)

    output_model_folder = data_util.create_model_folder()
    with open(output_model_folder + MAPPING_EMBEDDING_FILE, 'wb') as file:
        pickle.dump(mapping_embedding, file, protocol=pickle.HIGHEST_PROTOCOL)

    model = EntityLSTM(mapping_embedding.embedding_tensor, 32)

    model.run_train(100, output_model_folder, train_set, dev_set)
    
    #evaluate over test set
    print('Evaluate over test set')
    with tf.Session() as sess:
        transition_params = model.restore_pretrained_model(sess, output_model_folder)
        model.evaluate(sess, test_set, transition_params)
        

if __name__ == "__main__":
    main()
