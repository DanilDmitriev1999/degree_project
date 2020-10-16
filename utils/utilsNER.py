from tqdm import tqdm
import numpy as np
import collections
from pprint import pprint
import seaborn as sns
import matplotlib.pyplot as plt
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, RandomSampler, TensorDataset

from seqeval.metrics import f1_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Read_data:
    def __init__(self, path, percent=100):
        entries = open(path, "r").read().strip().split("\n\n")

        self.sentence, self.label = [], []  # list of lists
        for entry in entries:
            words = [line.split()[0] for line in entry.splitlines()]
            tags = [self._check(line.split()[-1]) for line in entry.splitlines()]
            self.sentence.append(words)
            self.label.append(tags)

        len_data = int(len(self.sentence) * percent / 100)
        self.sentence = [i for i in self.sentence[:len_data] if len(i) < 50 and len(i) > 5]
        self.label = [i for i in self.label[:len_data] if len(i) < 50 and len(i) > 5]


    def count_data(self, _chek_0=True):
        count_labels = collections.Counter()
        if _chek_0:
            train_label = [i for t in self.label for i in t]
        else:
            train_label = [i for t in self.label for i in t if i != 'O']
        for label in train_label:
           count_labels[label] += 1
        
        return count_labels

    def plot_count(self):
        count_labels = self.count_data(_chek_0=False)
        plt.figure(figsize=(15, 9))
        sns.set(style="darkgrid")

        x = list(count_labels.keys())
        y = list(count_labels.values())
        sns.barplot(x=x, y=y)

    def plot_len_data(self):
        plt.figure(figsize=(20, 6))
        len_sen = [len(i) for i in self.sentence]
        sns.distplot(len_sen)


    @staticmethod
    def _check(tag):
        if tag == 'B-Location':
            tag = 'B-LOC'
        if tag == 'I-Location':
            tag = 'I-LOC'
        if tag == 'B-Facility':
            tag = 'O'   
        if tag == 'I-Facility':
            tag = 'O'
        if tag == 'I-LocOrg':
            tag = 'I-LOC'
        if tag == 'B-LocOrg':
            tag = 'B-LOC' 
        if tag == 'B-MISC':
            tag = 'O'
        if tag == 'I-MISC':
            tag = 'O'
        if tag == 'B-Person':
            tag = 'B-PER'
        if tag == 'I-Person':
            tag = 'I-PER'
        if tag == 'B-Org':
            tag = 'B-ORG'
        if tag == 'I-Org':
            tag = 'I-ORG'
        if tag == 'B-Loc':
            tag = 'B-LOC'
        if tag == 'I-Loc':
            tag = 'I-LOC'
        return tag

    def __len__(self):
        return len(self.sentence)

    def __getitem__(self, i):
        return self.sentence[i], self.label[i]
    
def convert_tokens_to_ids(tokens, tokenizer, pad=True, ):
    max_len = 150
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    ids = torch.LongTensor([token_ids])
    assert ids.size(1) < max_len, print(ids.size(1))
    if pad:
        padded_ids = torch.zeros(max_len).long()
        padded_ids[:ids.size(1)] = ids
        mask = torch.zeros(max_len).long()
        mask[0:ids.size(1)] = 1
        return padded_ids, mask
    else:
        return ids

def subword_tokenize(tokens, labels, tokenizer):
    def flatten(list_of_lists):
        for list in list_of_lists:
            for item in list:
                yield item
                
    
    SEP = "[SEP]"
    MASK = "[MASK]"
    CLS = "[CLS]"
    subwords = list(map(tokenizer.tokenize, tokens))
    subword_lengths = list(map(len, subwords))
    subwords = [CLS] + list(flatten(subwords)) + [SEP]
    token_start_idxs = 1 + np.cumsum([0] + subword_lengths[:-1])
    # X label described in Bert Paper section 4.3
    bert_labels = [[label] + (sublen-1) * ["X"] for sublen, label in zip(subword_lengths, labels)]
    bert_labels = ["O"] + list(flatten(bert_labels)) + ["O"]

    assert len(subwords) == len(bert_labels)
    assert len(subwords) <= 512
    return subwords, token_start_idxs, bert_labels


def subword_tokenize_to_ids(tokens, labels, tokenize):
    assert len(tokens) == len(labels)
    subwords, token_start_idxs, bert_labels = subword_tokenize(tokens, labels, tokenize)
    subword_ids, mask = convert_tokens_to_ids(subwords, tokenizer)
    token_starts = torch.zeros(max_len)
    token_starts[token_start_idxs] = 1
    bert_labels = [labels_to_ids[label] for label in bert_labels]
    # X label described in Bert Paper section 4.3 is used for pading
    padded_bert_labels = torch.ones(max_len).long() * labels_to_ids["X"]
    padded_bert_labels[:len(bert_labels)] = torch.LongTensor(bert_labels)

    mask.require_grad = False
    return {
        "input_ids": subword_ids,
        "attention_mask": mask,
        "bert_token_starts": token_starts,
        "labels": padded_bert_labels
    }

def prepare_dataset(path, tokenize, percent=100):
    dataset = Read_data(path, percent)
    featurized_sentences = []
    for tokens, labels in dataset:
        features = subword_tokenize_to_ids(tokens, labels, tokenize)
        featurized_sentences.append(features)

    def collate(featurized_sentences_batch):
        keys = ("input_ids", "attention_mask", "bert_token_starts", "labels")
        output = {key: torch.stack([fs[key] for fs in featurized_sentences_batch], dim=0) for key in keys}
        return output

    dataset = collate(featurized_sentences)
    return TensorDataset(*[dataset[k] for k in ("input_ids", "attention_mask", "labels")])

def f1(outputs, ids_to_labels, labels_to_ids, average='Macro'):
    y_true, y_pred = [], []
    for out in outputs:
        batch_pred = out['y_pred'].cpu().numpy().tolist()
        batch_y = out['y_true'].cpu().numpy().tolist()
        batch_seq = out['mask'].cpu().numpy().sum(-1).tolist()
        for i, length in enumerate(batch_seq):
            batch_pred[i] = batch_pred[i][:length]
            batch_y[i] = batch_y[i][:length]
        y_true += batch_y
        y_pred += batch_pred

    flatten = lambda l: [item for sublist in l for item in sublist]
    y_true = flatten(y_true)
    y_pred = flatten(y_pred)
    y_true = [ids_to_labels[l] for l in y_true]
    y_pred = [ids_to_labels[l] for l in y_pred]

    ids = [i for i, label in enumerate(y_true) if label != "X"]
    y_true_cleaned = [y_true[i] for i in ids]
    y_pred_cleaned = [y_pred[i] for i in ids]

    f1_s = f1_score(y_true_cleaned, y_pred_cleaned, average=average)
    return f1_s

def do_epoch(model, criterion, data, ids_to_labels, labels_to_ids, 
             optimizer=None, name=None):
    epoch_loss = 0
    epoch_f1 = 0

    output = {
        'y_true': None,
        'y_pred': None,
        'mask': None,
              }

    outputs = []
    
    is_train = not optimizer is None
    name = name or ''
    model.train(is_train)
    
    batches_count = len(data)
    
    with torch.autograd.set_grad_enabled(is_train):
        with tqdm(total=batches_count) as progress_bar:
            for batch in data:
                text = batch[0].to(device)
                tags = batch[2].to(device)
                mask = batch[1].to(device)

                predictions = model(text, mask)

                X = labels_to_ids["X"]

                not_X_mask = tags != X
                active_loss = not_X_mask.view(-1)
                active_logits = predictions.view(-1, num_labels)
                active_labels = torch.where(
                        active_loss, tags.view(-1), torch.tensor(criterion.ignore_index).type_as(tags))
                loss = criterion(active_logits, active_labels)

                predict = torch.argmax(predictions, dim=-1)

                output.update({
                    'y_true': tags,
                    'y_pred': predict,
                    'mask': mask,
                })

                outputs.append(output)
                
                f1_s = f1(outputs, ids_to_labels, labels_to_ids)

                outputs = []

                epoch_loss += loss.item()
                epoch_f1 += f1_s

                if optimizer:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                progress_bar.update()
                progress_bar.set_description('{:>5s} Average Loss = {:.5f}, F1-score = {:.3%}'.format(
                    name, epoch_loss / batches_count, f1_s))

            progress_bar.set_description('{:>5s} Average Loss = {:.5f}, F1-score = {:.3%}'.format(
                name, epoch_loss / batches_count, epoch_f1 / batches_count))

    return epoch_loss / batches_count


def fit(model, criterion, optimizer, train_data, ids_to_labels, labels_to_ids,
        epochs_count=1, val_data=None):
    for epoch in range(epochs_count):
        name_prefix = '[{} / {}] '.format(epoch + 1, epochs_count)
        train_loss = do_epoch(model, criterion, train_data, ids_to_labels,
                               labels_to_ids, optimizer, name_prefix + 'Train:')

        if not val_data is None:
            val_loss = do_epoch(model, criterion, val_data, ids_to_labels,
                                 labels_to_ids, None, name_prefix + '  Val:')
            
