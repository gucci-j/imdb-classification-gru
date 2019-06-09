# from src.model import Model
from src.model_with_self_attention import Model
from src.load_data import load_data

import torch.nn as nn
import torch.optim as optim
import torch
# import spacy
# nlp = spacy.load('en')

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

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
    ATTN_FLAG = True

    print('data loading done')

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
        train_loss, train_acc = train_run(model, train_iterator, optimizer, criterion, ATTN_FLAG)
        valid_loss, valid_acc = eval_run(model, valid_iterator, criterion, ATTN_FLAG)
        print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}% |')
    
    test_loss, test_acc = eval_run(model, test_iterator, criterion, ATTN_FLAG)
    print(f'| Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}% |')
    if ATTN_FLAG is True:
        attn_visualization(model, test_iterator, TEXT, multiple_flag=True)
    torch.save(model.state_dict(), PATH)


def train_run(model, iterator, optimizer, criterion, aflag):
    epoch_loss = 0
    epoch_acc = 0
    model.train()

    if aflag is True:
        for index, batch in enumerate(iterator):
            print(f'Now: {index} / {len(iterator)}')
            optimizer.zero_grad()
            output, _ = model(batch.text)
            loss = criterion(output, batch.label)
            acc = binary_accuracy(output, batch.label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    else:
        for index, batch in enumerate(iterator):
            print(f'Now: {index} / {len(iterator)}')
            optimizer.zero_grad()
            output = model(batch.text).squeeze(1)
            loss = criterion(output, batch.label)
            acc = binary_accuracy(output, batch.label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def eval_run(model, iterator, criterion, aflag):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()

    with torch.no_grad():
        if aflag is True:
            for batch in iterator:
                predictions, _ = model(batch.text)
                loss = criterion(predictions, batch.label)
                acc = binary_accuracy(predictions, batch.label)

                epoch_loss += loss.item()
                epoch_acc += acc.item()
        else:
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

    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)

    return acc


def attn_visualization(model, iterator, TEXT, multiple_flag=False):
    """
    Visualize self-attention weights with input captions.
    """

    if multiple_flag is False:
        with torch.no_grad():
            batch = next(iter(iterator))
            _, attention = model(batch.text)

            # in torchtext, batch_size is placed in dim=1. dim=0 is used for sentence length
            text = batch.text.transpose(0, 1)
            # print(attention.size())
            attention_weight = attention.cpu().numpy()

            itos = []
            for text_element in text:
                itos_element = []
                for index in text_element:
                    # print(f'{TEXT.vocab.itos[index]} ')
                    itos_element.append(TEXT.vocab.itos[index])
                itos.append(itos_element)

            plt.figure(figsize = (16, 5))
            sns.heatmap(attention_weight, annot=np.asarray(itos), fmt='', cmap='Blues')
            plt.savefig('attention.png')

    elif multiple_flag is not False:
        with torch.no_grad():
            batch_count = 0
            for batch in iterator:
                _, attention = model(batch.text)
                text = batch.text.transpose(0, 1)
                attention_weight = attention.cpu().numpy()
                
                itos = []
                for text_element in text:
                    itos_element = []
                    for index in text_element:
                        itos_element.append(TEXT.vocab.itos[index])
                    itos.append(itos_element)
                
                fig_size = len(batch.text) + 1 # for changing fig_size dynamically
                plt.figure(figsize = (fig_size, 7))
                sns.heatmap(attention_weight, annot=np.asarray(itos), fmt='', cmap='Blues')
                plt.savefig('./fig/attention_' + str(batch_count) + '.png')
                plt.close()

                if batch_count == 10:
                    break
                batch_count += 1

if __name__ == "__main__":
    main()