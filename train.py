# from model import Model
from model_with_self_attention import Model
from load_data import load_data

import torch.nn as nn
import torch.optim as optim
import torch
# import spacy
# nlp = spacy.load('en')

def main():
    BATCH_SIZE = 32
    TEXT, LABEL, train_iterator, valid_iterator, test_iterator = load_data(BATCH_SIZE)
    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 512
    OUTPUT_DIM = 1
    NUM_LAYERS = 3
    DROPOUT = 0.4
    N_EPOCHS = 5
    PATH = './weight/weight_w_attention.pth'

    model = Model(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, DROPOUT)

    # check the embedding vector
    # pretrained_embeddings = TEXT.vocab.vectors
    # print(pretrained_embeddings.shape)

    # set an optimizer and a loss function
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()

    # for a gpu environment
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = criterion.to(device)

    for epoch in range(N_EPOCHS):
        train_loss, train_acc = train_run(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = eval_run(model, valid_iterator, criterion)
        print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}% |')
    
    test_loss, test_acc = eval_run(model, test_iterator, criterion)
    print(f'| Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}% |')

    torch.save(model.state_dict(), PATH)


def train_run(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()

    for batch in iterator:
        optimizer.zero_grad()
        output = model(batch.text).squeeze(1)
        loss = criterion(output, batch.label)
        acc = binary_accuracy(output, batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def eval_run(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
    
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum()/len(correct)

    return acc

if __name__ == "__main__":
    main()