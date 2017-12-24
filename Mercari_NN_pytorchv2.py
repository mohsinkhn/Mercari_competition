from __future__ import print_function, unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas(desc='progress')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import os
#import pickle
#import hashlib
import string
import unicodedata
import re
import gc
import time
import math

#import matplotlib.pyplot as plt
#import seaborn as sns

os.environ['OMP_NUM_THREADS'] = '4'

from collections import defaultdict, OrderedDict, Counter
#from nltk.corpus import stopwords
#from spacy.lang.en.stop_words import STOP_WORDS
from itertools import chain
num_partitions = 4
num_cores = 4
from multiprocessing import Pool, cpu_count
# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
stop_words = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 
              'there', 'about', 'once', 'during', 'out', 'very', 'having', 
              'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 
              'its', 'yours', 'such', 'into', 'most', 'itself', 'other', 
              'off', 'is', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 
              'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 
              'through', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 
              'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 
              'ours', 'had', 'she', 'all', 'when', 'at', 'any', 'before', 'them',
              'same', 'and', 'been', 'have', 'in', 'will', 'does', 'yourselves', 
              'then', 'that', 'because', 'what', 'over', 'why’, ‘so', 'can', 'did',
              'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only',
              'myself', 'which', 'those', 'i','after', 'few', 'whom', 'being', 'if', 
              'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']

def unicodeToAscii(s):
    return  unicodedata.normalize('NFKC', s)

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"'", r"", s)
    s = re.sub(r"[.!?':;,]", r" ", s)
    s = re.sub(r"-", r"", s)
    s = re.sub(r"[^0-9a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"0", r"zero", s)
    s = re.sub(r"1", r"one", s)
    s = re.sub(r"2", r"two", s)
    s = re.sub(r"3", r"three", s)
    s = re.sub(r"4", r"four", s)
    s = re.sub(r"5", r"five", s)
    s = re.sub(r"6", r"six", s)
    s = re.sub(r"7", r"seven", s)
    s = re.sub(r"8", r"eight", s)
    #s = re.sub(r"/s/s", r"/s", s)
    return s

def _normalize_and_ngrams(sent, ngrams):
    input_list = normalizeString(sent).split()
    input_list = [word for word in input_list if word not in stop_words]
    s = input_list.copy()
    for i in range(2, ngrams+1):
        s += [' '.join(input_list[j:j+i]) for j in range(len(input_list)-i + 1)]
        #s += list((zip(*[input_list[j:] for j in range(i)])))
    return s

#tmp = "I am not a dance'r and i am a 6ixy   c-o:d;er programmer"
#print(normalizeString(tmp))
#print(_normalize_and_ngrams(tmp, 3))

class Vocab_topwords():
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.index2word = {}
        
    def fit_data(self, data, col, ngrams=3, max_features=50000):
        c = Counter(list(chain.from_iterable(data[col].tolist())))
        for i, (w, count) in enumerate(c.most_common(max_features)):
            self.word2index[w] = i
        return
    

            
            
def prepareVocab(name, data, max_features):
    vocab = Vocab_topwords(name)
    vocab.fit_data(data, name, max_features=max_features)
    
    print("Counted words:")
    print(vocab.name, len(vocab.word2index))
    return vocab

def indexesFromSentence(vocab, tokens, ngrams, max_len):
    num_list = []
    for i, item in enumerate(tokens):
        if len(num_list) == max_len:
            break
        elif item in vocab.word2index:
            num_list.append(vocab.word2index[item])
        else:
            continue
        
    if len(num_list) < max_len :
        num_list += [0]*(max_len - len(num_list) )
        
    return num_list

def parallelize_dataframe(df, func):
    df_split = np.array_split(df, num_partitions)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def get_cat_1(x): return str(x).split('/')[0]
def get_cat_2(x): return str(x).split('/')[1] if len(str(x).split('/')) > 1 else -1
def get_cat_3(x): return ' '.join(str(x).split('/')[2:]) if len(str(x).split('/')) > 2 else -1

def applycat1(df): 
    df['cat1'] = df['category_name'].progress_apply(get_cat_1)
    return df

def applycat2(df): 
    df['cat2'] = df['category_name'].progress_apply(get_cat_2)
    return df

def applycat3(df): 
    df['cat3'] = df['category_name'].progress_apply(get_cat_3)
    return df

def norm3grams(s): return _normalize_and_ngrams(s, 3)

def applyname(series): return series.progress_apply(norm3grams)

def index2sent1(x, name_vocab): return indexesFromSentence(name_vocab, x, 3, 10)

def name2index(series): return series.progress_apply(lambda x: index2sent1(x, name_vocab))

def norm2grams(s): return _normalize_and_ngrams(s, 1)

def applydesc(series):return series.progress_apply(norm2grams)

def index2sent2(x, desc_vocab): return indexesFromSentence(desc_vocab, x, 1, 80)

def desc2index(series): return series.progress_apply(lambda x: index2sent2(x, desc_vocab))

def read_data(in_path, out_path):
    if os.path.exists(os.path.join(out_path, 'train_2.pkl')) and os.path.exists(os.path.join(out_path, 'test_2.pkl')):
        train_data = pd.read_pickle(os.path.join(out_path, 'train_2.pkl'))
        test_data  = pd.read_pickle(os.path.join(out_path, 'test_2.pkl'))
        
        return train_data, test_data
    
    else:
        train_data = pd.read_table(os.path.join(in_path, 'train.tsv'))
        test_data  = pd.read_table(os.path.join(in_path, 'test.tsv'))
    
        train_rows = len(train_data)
        data = pd.concat([train_data, test_data], ignore_index=True)
        
        data['name'] = data['name'].astype(str)
        data['item_description'] = data['item_description'].astype(str)
        
        #ddata = dd.from_pandas(data, 4)

        
        data = applycat1(data)
        data = applycat2(data)
        data = applycat3(data)
        data.fillna(-1, inplace=True)
        cat_cols = ['category_name', 'brand_name', 'item_condition_id', 'cat1', 'cat2', 'cat3']
        print("Label enoding categoricals")
        for col in cat_cols:
            data[col] = LabelEncoder().fit_transform(data[col].astype(str)).astype(np.int32)
            
        print("Tokenizing text columns")
        data['name'] = parallelize_dataframe(data['name'], applyname)
        print("Preparing vocabs")
        global name_vocab
        name_vocab = prepareVocab('name', data[['name']], 100000)
        data['name'] = name2index(data['name'])
        del name_vocab
        
        print("Transforming text to sequences")
        data['item_description'] = applydesc(data['item_description'])
        global desc_vocab
        desc_vocab = prepareVocab('item_description', data[['item_description']], 100000)
        data['item_description'] = desc2index(data['item_description'])
        del desc_vocab
        
        train_data = data.loc[: train_rows - 1, :]
        train_data = train_data.loc[(train_data.price >= 3) & (train_data.price <= 2000), :].reset_index(drop=True)
        test_data  = data.loc[train_rows: , :].reset_index(drop=True)
        
        del train_data['test_id']
        del test_data['train_id']
        del data
        gc.collect()
        print("Writing out new pickles dataframes")
        train_data.to_pickle(os.path.join(out_path, 'train_2.pkl'))
        test_data.to_pickle(os.path.join(out_path, 'test_2.pkl'))
        
        return
    
    
class ToTensor():
    """Convert numpy arrays to tensors"""
    def __call__(self, sample):
        
        item_name = sample['item_name']
        text_description = sample['text_description']
        brand_name = sample['brand_name']
        category = sample['category']
        item_condition_id = sample['item_condition_id']
        shipping_flag = sample['shipping_flag']
        cat1 = sample['cat1']
        cat2 = sample['cat2']
        cat3 = sample['cat3']
        target = sample['target']
        
        return {
                'item_name': torch.from_numpy(np.asarray(item_name)).long().view(-1), 
                'text_description': torch.from_numpy(np.asarray(text_description)).long().view(-1), 
                'brand_name': torch.from_numpy(np.asarray(brand_name)).long(), 
                'category': torch.from_numpy(np.asarray(category)).long(),
                'item_condition_id': torch.from_numpy(np.asarray(item_condition_id)).long(), 
                'shipping_flag': torch.from_numpy(np.asarray(shipping_flag)).type(torch.FloatTensor),
                'cat1': torch.from_numpy(np.asarray(cat1)).long(),
                'cat2': torch.from_numpy(np.asarray(cat2)).long(),
                'cat3': torch.from_numpy(np.asarray(cat3)).long(),
                'target': torch.log1p(torch.from_numpy(np.asarray(target))).type(torch.FloatTensor)
               }

class MercariDataset(Dataset):
    """Mercari item price prediction dataset"""
    def __init__(self, df, transform=None):
        """
        Args:
            df = dataframe with required columns
            train_file : file name for training data
            test_file: file name for test data
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = df
        self.transform = transform
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item_name = [self.data['name'].iloc[idx]]
        #print(item_name.shape)
        text_description = [self.data['item_description'].iloc[idx]]
        brand_name = [self.data['brand_name'].iloc[idx]]
        category = [self.data['category_name'].iloc[idx]]
        cat1 = [self.data['cat1'].iloc[idx]]
        cat2 = [self.data['cat2'].iloc[idx]]
        cat3 = [self.data['cat3'].iloc[idx]]
        item_condition_id = [self.data['item_condition_id'].iloc[idx]]
        shipping_flag = [self.data['shipping'].iloc[idx]]
        target = [self.data['price'].iloc[idx]]
        
        sample = {'item_name':item_name,
                 'text_description': text_description,
                 'brand_name': brand_name,
                 'category': category,
                 'item_condition_id': item_condition_id,
                 'shipping_flag': shipping_flag,
                 'cat1': cat1,
                 'cat2': cat2,
                 'cat3': cat3,
                 'target': target}
        #print(sample.shape)
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
class MercariNet(nn.Module):
    def __init__(self, input_sizes):
        super(MercariNet, self).__init__()
        
        self.nameEmbedding = nn.EmbeddingBag(input_sizes[0][0], input_sizes[0][1])
        self.textEmbedding = nn.EmbeddingBag(input_sizes[1][0], input_sizes[1][1])
        self.brandEmbedding = nn.Embedding(input_sizes[2][0], input_sizes[2][1])
        self.categoryEmbedding = nn.Embedding(input_sizes[3][0], input_sizes[3][1])
        self.conditionEmbedding = nn.Embedding(input_sizes[4][0], input_sizes[4][1])
        self.cat1Embedding = nn.Embedding(input_sizes[5][0], input_sizes[5][1])
        self.cat2Embedding = nn.Embedding(input_sizes[6][0], input_sizes[6][1])
        self.cat3Embedding = nn.Embedding(input_sizes[7][0], input_sizes[7][1])
        
        all_dims_sum = sum([dim[1] for dim in input_sizes]) + 1
        self.bn1 = nn.BatchNorm1d(all_dims_sum)
        self.fc1 = nn.Linear(all_dims_sum, 150)
        self.bn2 = nn.BatchNorm1d(150)
        self.fc2 = nn.Linear(150, 1)
        
    def forward(self, inputs):
        item_name = inputs['item_name']
        text_description = inputs['text_description']
        brand_name = inputs['brand_name']
        category = inputs['category']
        item_condition_id = inputs['item_condition_id']
        shipping_flag = inputs['shipping_flag']
        cat1 = inputs['cat1']
        cat2 = inputs['cat2']
        cat3 = inputs['cat3']
        
        batch_size = item_name.size()[0]
        #print(item_name.size())
        nameEmbeds = self.nameEmbedding(item_name).view(batch_size, 1, -1)
        #nameEmbedsMean = torch.mean(nameEmbeds, dim=1)
        #print(nameEmbeds.size())
        textEmbeds = self.textEmbedding(text_description).view(batch_size, 1, -1)
        #textEmbedsMean = torch.mean(textEmbeds, dim=1)
        #print(textEmbeds.size())
        brandEmbeds = self.brandEmbedding(brand_name).view(batch_size, 1, -1)
        
        categoryEmbeds = self.categoryEmbedding(category).view(batch_size, 1, -1)
        
        conditionEmbeds = self.conditionEmbedding(item_condition_id).view(batch_size, 1, -1)
        
        cat1Embeds = self.cat1Embedding(cat1).view(batch_size, 1, -1)
        cat2Embeds = self.cat2Embedding(cat2).view(batch_size, 1, -1)
        cat3Embeds = self.cat3Embedding(cat3).view(batch_size, 1, -1)
        
        shipping_flag = shipping_flag.view(batch_size, 1, -1)
        #print(conditionEmbeds.size())
        #print(shipping_flag.size())
        #Concat all embeddings and input
        combined = torch.cat((nameEmbeds, textEmbeds, brandEmbeds, categoryEmbeds, 
                              conditionEmbeds,  shipping_flag, cat1Embeds, cat2Embeds, cat3Embeds), dim=2)
        combined = combined.view(batch_size, -1)
        x = self.bn1(combined)
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.leaky_relu(self.fc1(x))
        x = self.bn2(x)
        x = F.dropout(x, p=0.02, training=self.training)
        output = self.fc2(x)
        #x = F.leaky_relu(x)
        #output = F.dropout(x, p=0.05, training=self.training)
        return output
        
# Some Useful Time functions
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))
    
# Training model function that uses the dataloader to load the data by Batch
def train_model(model, criterion, optimizer, num_epochs=5, print_every = 100):
    start = time.time()

    best_acc = 0.0
    print_loss_total = 0  # Reset every print_every

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            num_batches = dataset_sizes[phase]/BATCH_SIZE
            #running_corrects = 0

            # Iterate over data.
            for i_batch, sample_batched in enumerate(mercari_dataloaders[phase]): 
            # get the inputs
                inputs = {k: Variable(v) for k,v in sample_batched.items() if k != 'target'}
                #inputs = {'name':Variable(sample_batched['name']), 
                #          'item_description':Variable(sample_batched['item_desc']), \
                #    'brand_name':Variable(sample_batched['brand_name']), \
                #    'cat_name':Variable(sample_batched['cat_name']), \
                #    'general_category':Variable(sample_batched['general_category']), \
                #    'subcat1_category':Variable(sample_batched['subcat1_category']), \
                #    'subcat2_category':Variable(sample_batched['subcat2_category']), \
                #    'item_condition':Variable(sample_batched['item_condition']), \
                #    'shipping':Variable(sample_batched['shipping'].float())}
                #print(inputs)
                prices = Variable(sample_batched['target'])   
                batch_size = len(sample_batched['shipping_flag'])   
                

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                #_, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, prices)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                print_loss_total += loss.data[0]
                #running_corrects += torch.sum(preds == labels.data)
                
                
                if (i_batch+1) % print_every == 0:
                    print_loss_avg = print_loss_total / print_every
                    print_loss_total = 0
                    #print (i_batch / num_batches, i_batch, num_batches)
                    print('%s (%d %d%%) %.4f' % (timeSince(start, i_batch / num_batches), \
                                                 i_batch, i_batch / num_batches*100, print_loss_avg))
                
                # I have put this just so that the Kernel will run and allow me to publish
                #if (i_batch) > 500:
                #    break

            epoch_loss = running_loss / num_batches
            #epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            
        print()

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # load best model weights
    #model.load_state_dict(best_model_wts)
    return model
    
def predict(model, test_loader):
    preds = []
    for batch in test_loader:
        inputs = {k: Variable(v) for k,v in batch.items() if k != 'target'}
        #prices = Variable(batch['target'])
        outputs = model(inputs).data.numpy()
        preds.extend(outputs)
    return preds
    
    
if __name__=="__main__":
    gc.collect()
    print("Preparing data")
    read_data("../input", "./")
    BATCH_SIZE = 2048
    train = pd.read_pickle("train_2.pkl")
    #test = pd.read_pickle("test_2.pkl")
    dtrain, dvalid = train_test_split(train, test_size=0.1)
    print("loading datasets")
    mercari_datasets = {
                        'train': MercariDataset(dtrain,transform=transforms.Compose([ToTensor()])), 
                        'val': MercariDataset(dvalid,transform=transforms.Compose([ToTensor()]))
                       }
    dataset_sizes = {
               x: len(mercari_datasets[x]) for x in ['train', 'val']
                    }
    
    mercari_dataloaders = {
               'train': torch.utils.data.DataLoader(mercari_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
               'val': torch.utils.data.DataLoader(mercari_datasets['val'], batch_size=BATCH_SIZE, shuffle=True, num_workers=1)                                           
                    }
    
    mrnet = MercariNet([(100000, 100), (100000, 100), (6000, 30), (1600, 20), (5, 3), (15,4), (120, 10), (900, 20)])
    print(mrnet)
    optimizer = optim.Adam(mrnet.parameters(), lr=0.003, weight_decay=0)
    criterion = nn.MSELoss()
    train_model(mrnet,criterion,optimizer, 4)
    
    #del existing data loaders
    del mercari_datasets
    del mercari_dataloaders
    del train
    
    #Make test predictions
    test = pd.read_pickle("test_2.pkl")
    test_dataset =  MercariDataset(test,transform=transforms.Compose([ToTensor()]))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_preds = predict(mrnet, test_loader)
    
    print("Write out submission")
    submission: pd.DataFrame = test[['test_id']]
    submission['price'] = np.expm1(test_preds)
    submission.price = submission.price.clip(1, 2000)
    submission.to_csv("embedding_nn_v3.csv", index=False)


    
    
    


