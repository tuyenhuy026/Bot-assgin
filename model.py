import tensorflow as tf
from constants import *
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils import shuffle
import data_util
from data_util import Timer
import os
import pickle
from dataset import MappingAndEmbedding, Dataset

seed = 13
np.random.seed(seed)

class EntityLSTM:
    def __init__(self, embeddings, batch_size):
        self.model_name = 'entity_lstm'
        self.embeddings_tensor = embeddings
        self.batch_size = batch_size
        self.alphabet_size = len(VIETNAMESE_CHAR) + 4
        self.num_of_class = len(LABELS) - 1
        self._build()
    
    def _add_placeholders(self):
        """
        Add placeholders of models
        """
        self.labels = tf.placeholder(tf.int32, [None, None], 'y_true')
        self.label_lens = tf.placeholder(tf.int32, [None], 'lb_lens')
        self.word_ids = tf.placeholder(tf.int32, [None, None], 'word_ids')
        self.char_ids = tf.placeholder(tf.int32, [None, None, None], 'char_ids')
        self.word_lengths = tf.placeholder(tf.int32, [None, None], 'word_length')
        self.dropout_op = tf.placeholder(dtype=tf.float32, shape=[], name="dropout_op")
        self.is_training = tf.placeholder(tf.bool, name='phase')

    def _add_word_embeddings_layer(self):
        """
        Add word embedding and character embedding
        """
        #word embedding
        with tf.variable_scope('word_embedding'):
            word_embedding_weights = tf.Variable(self.embeddings_tensor,name='word_embedding_weights', dtype=tf.float32, trainable=False)
            word_embeddings = tf.nn.embedding_lookup(word_embedding_weights, self.word_ids, name='word_embeddings')

        #character embedding
        #contain 1 layer of character embedding then cocat with word embeddings
        with tf.variable_scope('character_embedding'):
            character_embedding_weights = tf.get_variable(name = 'character_embedding_weights', dtype=tf.float32, shape=[self.alphabet_size, CHAR_DIM])
            character_embeddings = tf.nn.embedding_lookup(character_embedding_weights, self.char_ids, name='character_embeddings')

            sh = tf.shape(character_embeddings)
            character_embeddings = tf.reshape(character_embeddings, shape=[-1, sh[-2], CHAR_DIM])
            word_lengths = tf.reshape(self.word_lengths, shape=[-1])

            cell_fw = tf.contrib.rnn.LSTMCell(CHAR_HIDDEN_STATE_DIM)
            cell_bw = tf.contrib.rnn.LSTMCell(CHAR_HIDDEN_STATE_DIM)

            _, ((_, output_fw), (_, output_bw)) = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                                  cell_bw,
                                                                                  character_embeddings,
                                                                                  sequence_length=word_lengths,
                                                                                  dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            # shape = (batch size, max sentence length, char hidden size)
            output = tf.reshape(output, shape=[-1, sh[1], 2 * CHAR_HIDDEN_STATE_DIM])

            word_embeddings = tf.concat([word_embeddings, output], axis=-1)

        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout_op)

    def _add_token_lstm_layer(self, initializer):
        """
        Add token biLSTM layer, feedfoward afterward
        """

        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(TOKEN_LSTM_HIDDEN_STATE_DIM, initializer=initializer)
            cell_bw = tf.contrib.rnn.LSTMCell(TOKEN_LSTM_HIDDEN_STATE_DIM, initializer=initializer)
            output, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                                    self.word_embeddings,
                                                                    sequence_length=self.label_lens,
                                                                    dtype=tf.float32)
            lstm_output1 = tf.concat(output, 2)
            lstm_output1 = tf.nn.dropout(lstm_output1, self.dropout_op)
            lstm_output1 = tf.layers.batch_normalization (lstm_output1,training= self.is_training)

        with tf.variable_scope("feedforward_after_lstm"):
            W = tf.get_variable("W", shape=[2 * TOKEN_LSTM_HIDDEN_STATE_DIM, TOKEN_LSTM_HIDDEN_STATE_DIM],
                                initializer=initializer)
            b = tf.get_variable("b", shape=[TOKEN_LSTM_HIDDEN_STATE_DIM], dtype=tf.float32,
                                initializer=tf.zeros_initializer())

            ntime_steps = tf.shape(lstm_output1)[1]

            lstm_output1 = tf.reshape(lstm_output1, [-1, 2 * TOKEN_LSTM_HIDDEN_STATE_DIM])
            outputs = tf.nn.xw_plus_b(lstm_output1, W, b, name="output_before_tanh")
            outputs = tf.layers.batch_normalization (outputs,training= self.is_training)
            outputs = tf.nn.tanh(outputs, name="output_after_tanh")
            outputs = tf.nn.dropout(outputs, self.dropout_op)

        with tf.variable_scope("feedforward_before_crf"):
            W = tf.get_variable("W", shape=[TOKEN_LSTM_HIDDEN_STATE_DIM, self.num_of_class],
                                dtype=tf.float32, initializer=initializer)

            b = tf.get_variable("b", shape=[self.num_of_class], dtype=tf.float32,
                                initializer=tf.zeros_initializer())

            pred = tf.nn.xw_plus_b(outputs, W, b, name="output_before_crf")

            self.logits = tf.reshape(pred, [-1, ntime_steps, self.num_of_class])
            # self.logits = tf.nn.dropout(logits, self.dropout_op)
    
    def _add_loss(self, initializer):
        """
        Add loss function
        """
        self.transition_params = tf.get_variable('transition', shape=[self.num_of_class, self.num_of_class], initializer=initializer)

        log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(self.logits, self.labels, self.label_lens,
                                                              self.transition_params)
        self.loss = tf.reduce_mean(-log_likelihood)
    
    def _add_train_op(self):
        """
        Add train_op to self
        """
        self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.variable_scope("train_step"):
            tvars = tf.trainable_variables()
            grad, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 5.0)
            optimizer = tf.train.RMSPropOptimizer(learning_rate=5e-4, momentum=0.9)

            self.train_op = optimizer.apply_gradients(zip(grad, tvars))

    def _predict_step(self, sess, transition_params, word_ids, char_ids, label_ids, sequence_lengths, word_lengths):
        feed_dict = {self.word_ids: word_ids,
                     self.char_ids: char_ids,
                     self.labels: label_ids,
                     self.label_lens: sequence_lengths,
                     self.word_lengths: word_lengths,
                     self.dropout_op: 1.0,
                     self.is_training: False}
        unary_scores = sess.run(self.logits, feed_dict=feed_dict)
        y_pred = []
        y_true = []
        for unary_score, label, leng in zip(unary_scores, label_ids, sequence_lengths):
            unary_score = unary_score[:leng]
            label = label[:leng]
            pred_label, _ = tf.contrib.crf.viterbi_decode(unary_score, transition_params)
            y_pred.extend(pred_label)
            y_true.extend(label)
        return y_pred, y_true

    def evaluate(self, sess, evaluate_set, transition_params):
        print('Evaluating over {} dataset'.format(evaluate_set.name))

        all_y_pred = []
        all_y_true = []
        num_batch = len(evaluate_set.X) // self.batch_size + 1
        if(len(evaluate_set.X) % self.batch_size == 0): num_batch -= 1
        for idx, batch in enumerate(self._next_batch(evaluate_set.X, evaluate_set.Y, num_batch)):
            word_ids, char_ids, label_ids, sentence_lengths, word_lenghts = batch
            y_pred, y_true = self._predict_step(sess, transition_params, word_ids, char_ids, label_ids, sentence_lengths, word_lenghts)
            all_y_pred.extend(y_pred)
            all_y_true.extend(y_true)
        
        f1score = {}
        for f1_style in ['weighted', 'micro', 'macro']:
            f1score[f1_style] = f1_score(y_true, y_pred, average=f1_style) * 100
        accuracy = accuracy_score(y_true, y_pred) * 100
        for f1_style in ['weighted', 'micro', 'macro']:
            print('F1 score in mode {}: {}'.format(f1_style, f1score[f1_style]))
        print('Accuracy: {}'.format(accuracy))

        return f1score['macro']

    def _next_batch(self, data, label, numbatch):
        start = 0
        idx = 0
        while(idx < numbatch):
            X_batch = data[start:start + self.batch_size]
            Y_batch = label[start:start + self.batch_size]

            word_ids, char_ids = zip(*[zip(*x) for x in X_batch])
            word_ids, sentence_lengths = data_util.pad_sequences(word_ids, pad_tok=PADDING_TOKEN_INDEX)
            char_ids, word_lengths = data_util.pad_sequences(char_ids, PADDING_CHARACTER_INDEX, 2)
            label_ids, _ = data_util.pad_sequences(Y_batch, pad_tok=4)
            start += self.batch_size
            idx += 1
            yield(word_ids, char_ids, label_ids, sentence_lengths, word_lengths)
    
    def _build(self):
        self._add_placeholders()
        self._add_word_embeddings_layer()
        
        initializer = tf.contrib.layers.xavier_initializer()

        self._add_token_lstm_layer(initializer)
        self._add_loss(initializer)
        self._add_train_op()

    def _train_step(self, epochs):
        if not os.path.exists(MODELS_FOLDER):
            os.makedirs(MODELS_FOLDER)
        
        saver = tf.train.Saver(max_to_keep=2)
        best_f1 = 0.0
        nepoch_no_imp = 0
        
        session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=8,
        inter_op_parallelism_threads=8,
        log_device_placement=False
        )
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            num_batch_train = len(self.X_train) // self.batch_size + 1
            if(len(self.X_train) % self.batch_size == 0): num_batch_train -= 1
            for e in range(epochs):
                print('Epoch {}'.format(e+1))
                X_train, Y_train = shuffle(self.X_train, self.Y_train)
                for idx, batch in enumerate(self._next_batch(X_train, Y_train, num_batch_train)):
                    word_ids, char_ids, label_ids, sentence_lengths, word_lengths = batch
                    feed_dict = {self.word_ids: word_ids,
                                 self.char_ids: char_ids,
                                 self.labels: label_ids,
                                 self.label_lens: sentence_lengths,
                                 self.word_lengths: word_lengths,
                                 self.dropout_op: 0.5,
                                 self.is_training: True}

                    _, _, loss_train = sess.run([self.train_op, self.extra_update_ops, self.loss], feed_dict=feed_dict)
                    if(idx % 20 == 0):
                        print('Iter {}/{}, Loss = {}'.format(idx, num_batch_train, loss_train), end='\r', flush=True)

                print('End epoch {}, evaluating the model ...'.format(e + 1))
                
                transition_params = sess.run(self.transition_params)
                f1 = self.evaluate(sess, self.dev_set, transition_params)
                if(f1 > best_f1):
                    print('Accuracy improved, saving model at epoch {}.....'.format(e + 1))
                    saver.save(sess, self.model_folder + '/{}'.format(self.model_name))
                    print('Model was saved at {}'.format(self.model_folder))
                    best_f1 = f1
                    nepoch_no_imp = 0
                else:
                    nepoch_no_imp += 1
                    if(nepoch_no_imp > 10):
                        print('There is no improvement after 10 epochs, terminate...')
                        break

    def run_train(self, epochs, model_folder, train_set, dev_set):
        
        self.X_train = train_set.X
        self.Y_train = train_set.Y

        self.dev_set = dev_set
        self.model_folder = model_folder

        self._train_step(epochs)

    def restore_pretrained_model(self, sess, model_folder):
        #restore model
        print('Restore pretrained models at {}'.format(model_folder))
        saver = tf.train.Saver()
        saver.restore(sess, model_folder + '/{}'.format(self.model_name))
        
        transition_params = sess.run(self.transition_params)

        return transition_params

    def predict(self, sess, transition_params, mapping_embedding, text, core_nlp = None):
        timer = Timer()
        
        timer.start('Preprocess')
        if(core_nlp != None):
            token_sequences, token_position_sequences, fake_label_sequences, total_num_token = data_util.preprocess_text_with_tokenizer(text, core_nlp)
        else:
            token_sequences, fake_label_sequences,  total_num_token = data_util.preprocess_text(text)
        timer.stop()

        timer.start('Convert to dataset')
        temp_dataset = Dataset('deploy_text', token_sequences, fake_label_sequences, mapping_embedding)
        timer.stop()

        timer.start('Predict')
        num_batch = len(temp_dataset.X) // self.batch_size + 1
        if(len(temp_dataset.X) % self.batch_size == 0): num_batch -= 1
        predict_labels = []
        for indx, batch in enumerate(self._next_batch(temp_dataset.X, temp_dataset.Y, num_batch)):
            word_ids, char_ids, label_ids, sequence_lengths, word_lenghths = batch
            y_pred, _ = self._predict_step(sess, transition_params, word_ids, char_ids, label_ids, sequence_lengths, word_lenghths)
            predict_labels.extend(y_pred)
        timer.stop()

        timer.start('Prepare ouput')
        assert(total_num_token == len(predict_labels))
        result = []
        entity_tokens = []
        token_index = 0
        previous_label = 'O'
        for token_sequence in token_sequences:
            for token in token_sequence:
                label = LABELS[predict_labels[token_index]]
                token_index += 1
                if(label == 'O'):
                    if(previous_label == 'O'): 
                        previous_label = label
                        continue
                    else:
                        if(len(entity_tokens) > 0): 
                            result.append((entity_tokens, previous_label))
                            entity_tokens = []
                            previous_label = label
                else:
                    if(previous_label != label):
                        if(len(entity_tokens) > 0): result.append((entity_tokens, previous_label))
                        entity_tokens = []
                        entity_tokens.append(token)
                        previous_label = label
                    else:
                        entity_tokens.append(token)
                        previous_label = label
        if(len(entity_tokens) > 0): result.append((entity_tokens, previous_label))
        # token_index = 0
        # result = ''
        # previous_end = -1
        # for token_sequence, token_position_sequence in zip(token_sequences, token_position_sequences):
        #     for token, token_position in zip(token_sequence, token_position_sequence):
        #         start, end = token_position
        #         original_word = ''
        #         if(start == previous_end):
        #             result = result[:len(result) - 1]
        #         for index in range(start, end):
        #             original_word += text[index]
        #         assert(original_word == token), 'Word Error ' + original_word + ' ' + word
        #         try:
        #             original_word += '/' + LABELS[predict_labels[token_index]] + text[end]
        #         except:
        #             original_word += '/' + LABELS[predict_labels[token_index]]
        #         result += original_word
        #         original_word = ''
        #         previous_end = end
        #         token_index += 1
        timer.stop()

        return result