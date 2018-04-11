from model import EntityLSTM
import pickle
import tensorflow as tf
from constants import *
from dataset import Dataset, MappingAndEmbedding

def main():
    #An example for deploy model
    
    #First we need to load enssential part of pretrained model
    sess = tf.Session()
    pretrained_model_folder = MODELS_FOLDER + '/vi.vec_100d_89_f1_macro'
    
    #load mapping and embedding
    with open(pretrained_model_folder + MAPPING_EMBEDDING_FILE, 'rb') as file:
        mapping_embedding = pickle.load(file)
    
    #init and restore model
    model = EntityLSTM(mapping_embedding.embedding_tensor, 32)
    #transition parameters for viterbi decode
    transition_params = model.restore_pretrained_model(sess, pretrained_model_folder)
    
    # From here call predict function to get the result
    # The input here is "text"
    # The result return from predict function is a list of entities contained in the input 
    # Each entity in the list is stored as a pair, the 1st element is a list contain tokens of the entity,
    # 2nd element is the label of that entity.
    # For example: if the input is "Hôm nay thời tiết Hà Nội đẹp quá, Donald Trump từ Mỹ đã viến thăm Việt Nam an toàn"
    # The output should be: [(['Hà', 'Nội'], 'LOC'), (['Donald', 'Trump'], 'PER'), (['Mỹ'], 'LOC'), (['Việt', 'Nam'], 'LOC')]

    #deploy loop for several doc
    while(True):
        text = input('Input text: ')
        result = model.predict(sess, transition_params, mapping_embedding, text)
        print(result)
        print(type(result))

    #close the session before quit 
    sess.close()

if __name__ == "__main__":
    main()