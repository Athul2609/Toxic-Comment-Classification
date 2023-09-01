import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 3e-5 
WEIGHT_DECAY = 5e-4 
BATCH_SIZE = 32
NUM_EPOCHS = 100
NUM_WORKERS = 2
EMBEDDING_DIMENSION=50
EMBEDDER="GLOVE"
MAX_SEQ_LENGTH=256
HIDDEN_SIZE=64
CHECKPOINT_FILE = "best.pth.tar" 
PIN_MEMORY = True 
SAVE_MODEL = True
LOAD_MODEL = True
EMBEDDING_MATRIX_PATH=r"C:\Users\athul\myfiles\projects\toxic comment classifier\embeddings\embedding_matrix_50.npy"
WORD2IDX_PATH=r"C:\Users\athul\myfiles\projects\toxic comment classifier\embeddings\word2idx_50.pkl"