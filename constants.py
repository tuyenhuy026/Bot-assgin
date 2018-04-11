WORD_EMBEDDING_FILEPATH = 'word_vectors/vi.vec'

DATA_FOLDER = 'data'
MODELS_FOLDER = 'models'
TRAIN_DATA = DATA_FOLDER + '/train.txt'
DEV_DATA = DATA_FOLDER + '/dev.txt'
TEST_DATA = DATA_FOLDER + '/test.txt'
MAPPING_EMBEDDING_FILE = '/mapping&embedding.pickle'

#mode
TRAIN = 0
DEV = 1
TEST = 2

MODE = ['train', 'dev', 'test']

PAD_TOKEN = '$PAD_TOKEN$'
UNK = '$UNK$'
UNK_TOKEN_INDEX = 1
PADDING_TOKEN_INDEX = 0

LABELS = ['ORG', 'PER', 'LOC', 'O', 'P']

PADDING_CHARACTER_INDEX = 0
PAD_CHARACTER = '$CHAR_PAD$'
DIGIT_INDEX = 1
DIGIT = '$DIGIT$'
PUNT_INDEX = 2
PUNT = '$PUNT$'
DIGITS = '0123456789'
PUNTS = ',.!?<>{}[]()\"\\\'@#$%^&*_+-=~`'
UNK_CHAR_INDEX = 3
UNK_CHAR = '$UNK_CHAR$'

VIETNAMESE_CHAR = 'aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆ\
fFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTu\
UùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ'

END_PUNT = '.?!'
MIDDLE_PUNT = ',\'\"(){}[]'

MAX_SENTENCE_LEN = 256

EMBEDDING_DIM = 100
TOKEN_LSTM_HIDDEN_STATE_DIM = 100
CHAR_DIM = 25
CHAR_HIDDEN_STATE_DIM = 25
