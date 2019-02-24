import torch
from torchtext import data, datasets
import random

def load_data(BATCH_SIZE=16):
    SEED = 1234

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    TEXT = data.Field(tokenize='spacy')
    LABEL = data.LabelField(dtype=torch.float)

    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
    train_data, valid_data = train_data.split(random_state=random.seed(SEED))

    TEXT.build_vocab(train_data, max_size=25000, vectors="glove.6B.300d")
    LABEL.build_vocab(train_data)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data), 
        batch_size=BATCH_SIZE, 
        device=device)
    
    return TEXT, LABEL, train_iterator, valid_iterator, test_iterator