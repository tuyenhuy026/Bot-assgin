import codecs
from constants import *
import data_util
import numpy as np
import pickle

class Dataset(object):

    def __init__(self, name, token_sequences, label_sequences, mapping_embedding):
        self.name = name
        self.token_sequences = token_sequences
        self.label_sequences = label_sequences
        self.token2index = mapping_embedding.token2index
        self.character2index = mapping_embedding.character2index
        self._convert_to_indices()
    
    def _convert_to_indices(self):
        # print('Convert ' + self.name + ' data to indices')

        token_sequences = self.token_sequences
        label_sequences = self.label_sequences
        X = []
        Y = []

        for token_sequence, label_sequence in zip(token_sequences, label_sequences):
            token_and_char_ids_sequence = []
            label_id_sequence = []
            for token, label in zip(token_sequence, label_sequence):
                token_and_char_ids = self._get_word_and_char_ids(token)
                label_id = LABELS.index(label)
                
                token_and_char_ids_sequence.append(token_and_char_ids)
                label_id_sequence.append(label_id)
            X.append(token_and_char_ids_sequence)
            Y.append(label_id_sequence)

        self.X = X
        self.Y = Y

    def _get_word_and_char_ids(self, word):
        #return word's id and ids of the chars it contains
        vocabulary = self.token2index.keys()
        char_ids = []
        for char in word:
            if(char in VIETNAMESE_CHAR): char_ids.append(self.character2index[char])
            elif(char in PUNTS): char_ids.append(PUNT_INDEX)
            elif(char in DIGITS): char_ids.append(DIGIT_INDEX)
            else: char_ids.append(UNK_CHAR_INDEX)
        if(word in vocabulary): word_id = self.token2index[word]
        elif(word.lower() in vocabulary): word_id = self.token2index[word.lower()]
        else:
            word_id = UNK_TOKEN_INDEX
        return word_id, char_ids

class MappingAndEmbedding(object):

    def __init__(self, token2vec):
        print('Load pretrained token embedding to tensor and mapping....')
        token2index = {}
        character2index = {}
        vocabulary = token2vec.keys()

        token2index[UNK] = UNK_TOKEN_INDEX
        token2index[PAD_TOKEN] = PADDING_TOKEN_INDEX

        for indx, word in enumerate(vocabulary):
            token2index[word] = indx + 2

        character2index[PAD_CHARACTER] = PADDING_CHARACTER_INDEX
        character2index[UNK_CHAR] = UNK_CHAR_INDEX
        character2index[DIGIT] = DIGIT_INDEX
        character2index[PUNT] = PUNT_INDEX

        for indx, char in enumerate(VIETNAMESE_CHAR):
            character2index[char] = indx + 4

        embedding_tensor = np.zeros([len(vocabulary) + 2, EMBEDDING_DIM])
        for word in vocabulary:
            try:
                embedding_tensor[token2index[word]] = token2vec[word]
            except:
                print('Error at word: ' + word)
                pass
        
        self.token2index = token2index
        self.character2index = character2index
        self.embedding_tensor = embedding_tensor
        print('Embedding shape: ', np.shape(embedding_tensor))
        print('Token index size: ', len(token2index.keys()))
        print('Character index size: ', len(character2index.keys()))