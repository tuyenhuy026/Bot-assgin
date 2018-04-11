import codecs
from constants import *
import numpy as np
import re
import os
import time
import datetime

class Timer(object):

    def start(self, s):
        self.st = time.time()
        self.s = s
    def stop(self):
        self.end = time.time()
        print(self.s + ' time: ' + str(self.end - self.st))

def load_token_embeddings():
    print('Loading token embeddings from pretrained word vectors... ')
    file_input = codecs.open(WORD_EMBEDDING_FILEPATH, 'r', 'utf-8')
    token2vec = {}
    
    for line in file_input:
        line = line.strip().split(' ')
        if len(line) == 0: continue
        token = line[0]
        vector = np.array([float(x) for x in line[1:]])
        token2vec[token] = vector
    file_input.close()
    return token2vec

def load_raw_data():
    print('Loading raw data....')
    # all_word = []
    # all_character = []
    token_sequences = {}
    label_sequences = {}
    for mode, filepath in enumerate([TRAIN_DATA, DEV_DATA, TEST_DATA]):
        token_sequences[mode] = []
        label_sequences[mode] = []
        with codecs.open(filepath, 'r', 'utf-8') as file:
            print('Reading file for ' + MODE[mode] +' set')
            new_token_seq = []
            new_label_seq = []
            index = 0
            for line in file:
                print(index, end='\r', flush=True)
                index += 1                
                line = line.strip().split('\t')
                if len(line) == 0 or len(line[0]) == 0 or line[0] in '.?': #end of a sentence
                    if(len(new_label_seq)):
                        token_sequences[mode].append(new_token_seq)
                        label_sequences[mode].append(new_label_seq)
                        new_token_seq = []
                        new_label_seq = []
                    continue
                label = line[-1]
                token = line[0]
                new_label_seq.append(label)
                new_token_seq.append(token)
                #add word and character to vocab
                # if(token not in all_word): all_word.append(token)
                # for char in token:
                #     if(char not in all_character): all_character.append(char)
            if(len(new_label_seq) > 0):
                token_sequences[mode].append(new_token_seq)
                label_sequences[mode].append(new_label_seq)
            print()
    return token_sequences, label_sequences #, all_word, all_character

def create_model_folder():
    time_stamp = time.time()
    time_string = datetime.datetime.fromtimestamp(time_stamp).strftime('%m-%d-%Y_%H-%M-%S')
    word_vector_set_name = WORD_EMBEDDING_FILEPATH.split('/')[-1]
    model_folder = MODELS_FOLDER + '/' + time_string + '_' +  word_vector_set_name + '_' + str(EMBEDDING_DIM)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    
    return model_folder

def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: token to pad
    Return:
        a list of list where each sub list has the same length
    """
    padded_sequences, sequence_lengths = [], []
    for seq in sequences:
        seq = list(seq)
        padded_seq = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        padded_sequences += [padded_seq]
        sequence_lengths += [min(len(seq), max_length)]
    return padded_sequences, sequence_lengths

def pad_sequences(sequences, pad_tok, level = 1):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    if(level == 1):
        max_length = max(map(lambda x: len(x), sequences))
        padded_sequences, sequence_lengths = _pad_sequences(sequences, pad_tok, max_length)
    elif(level == 2): #padding level character
        #padding character to word so that all word in the sentence is the same length:
        max_length_word = max([max(map(lambda x: len(x), seq)) for seq in sequences])
        sentence_with_padded_words, word_lengths_per_sentence = [], []
        for sentence in sequences:
            padded_words, word_lengths = _pad_sequences(sentence, pad_tok, max_length_word)
            sentence_with_padded_words += [padded_words]
            word_lengths_per_sentence += [word_lengths]

        max_length_sentence = max(map(lambda x: len(x), sequences))
        #padding a max length word with all padding char to the sentence needed to be paded
        padded_sequences, _ = _pad_sequences(sentence_with_padded_words, [pad_tok] * max_length_word, max_length_sentence)
        #for the length of padding word will be 0
        sequence_lengths, _ = _pad_sequences(word_lengths_per_sentence, 0, max_length_sentence)        
    else:
        return None
    return padded_sequences, sequence_lengths

def _get_start_and_end_offset_of_token_from_spacy(token):
    start = token.idx
    end = start + len(token)
    return start, end

def preprocess_text_with_tokenizer(text, core_nlp):
    document = core_nlp(text)
    token_sequences = []
    fake_label_sequences = []
    token_position_sequences = []
    total_num_token = 0
    
    for span in document.sents:
        sentence = [document[i] for i in range(span.start, span.end)]
        token_sequence = []
        token_position_sequence = []
        fake_label_sequence = []
        for token in sentence:
            token_start, token_end = _get_start_and_end_offset_of_token_from_spacy(token)
            token_text = text[token_start:token_end]
            if(token_text.strip() in ['\n', '\t', ' ', '']):
                continue
            if len(token_text.split(' ')) != 1:
                print("WARNING: the text of the token contains space character, replaced with hyphen\n\t{0}\n\t{1}".format(token_text, 
                                                                                                                           token_text.replace(' ', '-')))
                token_text = token_text.replace(' ', '-')
            token_sequence.append(token_text)
            total_num_token += 1
            fake_label_sequence.append('O')
            token_position_sequence.append((token_start, token_end))
        token_sequences.append(token_sequence)
        fake_label_sequences.append(fake_label_sequence)
        token_position_sequences.append(token_position_sequence)

    return token_sequences, token_position_sequences, fake_label_sequences, total_num_token

def preprocess_text(text):
    pos = 0
    token = ''
    token_sequence = []
    token_sequences = []
    fake_label_sequence = []
    fake_label_sequences = []
    total_num_token = 0

    tokens = text.replace('\n', '. ').split(' ')
    while '' in tokens:
        tokens.remove('')
    for index, token in enumerate(tokens):
        if(len(token_sequence) > MAX_SENTENCE_LEN):
            token_sequences.append(token_sequence)
            fake_label_sequences.append(fake_label_sequence)
            token_sequence = []
            fake_label_sequence = []
        if len(token) == 0: continue
        elif len(token) == 1:
            token_sequence.append(token)
            fake_label_sequence.append('O')
            total_num_token += 1
            if(token in END_PUNT):
                if(index + 1 == len(tokens)):
                    if(len(token_sequence) > 0): 
                        token_sequences.append(token_sequence)
                        fake_label_sequences.append(fake_label_sequence)
                        token_sequence = []
                        fake_label_sequence = []
                elif(tokens[index + 1] not in END_PUNT):
                    if(len(token_sequence) > 0): 
                        token_sequences.append(token_sequence)
                        fake_label_sequences.append(fake_label_sequence)
                        token_sequence = []
                        fake_label_sequence = []
        else:
            pre_punt = []
            post_punt = []       
            end = False
            if(len(token) > 0):
                while token[0] in MIDDLE_PUNT or token[0] in END_PUNT:
                    pre_punt.append(token[0])
                    token = token[1:]
                    if(len(token) == 0): break
            if(len(token) > 0):
                while token[len(token) - 1] in MIDDLE_PUNT or token[len(token) - 1] in END_PUNT:
                    post_punt.append(token[len(token) - 1])
                    if(not end):
                        if(index + 1 < len(tokens)):
                            end = token[len(token) - 1] in END_PUNT and tokens[index + 1] not in END_PUNT
                        else:
                            end = token[len(token) - 1] in END_PUNT
                    token = token[:len(token) - 1]
                    if(len(token) == 0): break
            if(len(pre_punt) > 0): 
                token_sequence += pre_punt
                fake_label_sequence += ['O'] * len(pre_punt)
                total_num_token +=  len(pre_punt)
            if(len(token) > 0): 
                token_sequence.append(token)
                fake_label_sequence.append('O')
                total_num_token += 1
            if(len(post_punt) > 0): 
                token_sequence += post_punt
                fake_label_sequence += ['O'] * len(post_punt)
                total_num_token += len(post_punt)
            if(end):
                token_sequences.append(token_sequence) 
                fake_label_sequences.append(fake_label_sequence)
                token_sequence = []
                fake_label_sequence = []

    if(len(token_sequence) > 0):
        token_sequences.append(token_sequence)
        fake_label_sequences.append(fake_label_sequence)
    return token_sequences, fake_label_sequences, total_num_token