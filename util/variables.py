MODELS_PATH = "../output/models"
TESTS_PATH = "../output/tests"
VISUAL_PATH = "../output/visualization"
TABLES_PATH = f"{VISUAL_PATH}/tables"
GRAPHS_PATH = f"{VISUAL_PATH}/graphs"

ACC = 'accuracy'
PREC = 'precision'
REC = 'recall'
F1 = 'f1 score'
MODEL = 'model'
METHOD = 'method'
BI_VAR = 'bi'
WL_VAR = 'wl'
EBD_VAR = 'ebd'
OPT_VAR = 'opt'
EP_VAR = 'ep'
RD_VAR = 'rd'
D_VAR = 'd'
LR_VAR = 'lr'
L2_VAR = 'l2'

TRAIN_ACC = 'train accuracy'
TRAIN_LOSS = 'train loss'
TRAIN_F1 = 'train f1 score'
VALID_ACC = 'validation accuracy'
VALID_LOSS = 'validation loss'
VALID_F1 = 'validation f1 score'

RED_VAR = 'red'
CONT_VAR = 'cont'
WEIGHT_VAR = 'w'

BI = 'bidirectional'
WL = 'weighted loss'
EBD = 'embeddings function'
OPT = 'dataset optimization'
EP = 'epochs'
RD = 'reducer'
D = 'dropout'
LR = 'learning rate'
L2 = 'l2'

RED = 'reduction function'
CONT = 'speaker context'
RED_WEIGHT = 'reduction function weight generator'

UTT_ACC = 'utterance level accuracy'
A_ACC = 'speaker A final accuracy'
B_ACC = 'speaker B final accuracy'

LSTM = 'lstm'
DIALOGUERNN = 'dialogue'

LAST = 'last'
MOST = 'most'
WEIGHT = 'weight'
PATTERN = 'pattern'

MINI = 'mini'
MPNET = 'mpnet'

NONE = 'none'
SKIP_MOSTLY_ZERO = 'skip-mostly-zero'
SKIP_ZERO = 'skip-zero'

type_to_transformer = {
    MINI: 'paraphrase-MiniLM-L6-v2',
    MPNET: 'paraphrase-mpnet-base-v2'
}

transformer_to_input_size = {
    MINI: 384,
    MPNET: 768
}
