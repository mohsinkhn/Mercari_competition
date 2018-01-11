
__author__ = 'Mohsin Khan'


import copy
import numpy as np
import pandas as pd
import random
# import matplotlib.pyplot as plt
import sys
import os
import pickle
import hashlib
import string
import unicodedata
import re
from tqdm import tqdm, tqdm_notebook
tqdm.pandas()

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, QuantileTransformer
import lightgbm as lgb
from sklearn import metrics
import gc

from collections import defaultdict, OrderedDict, Counter
#from nltk.corpus import stopwords
#from spacy.lang.en.stop_words import STOP_WORDS
from itertools import chain

# from __future__ import print_function
np.random.seed(786)  # for reproducibility
from keras.models import Sequential, Model
from keras.layers import *
from keras.optimizers import RMSprop, Adam, Nadam
from keras.utils import np_utils 
from keras.layers.convolutional import Convolution1D, MaxPooling1D, ZeroPadding1D, AveragePooling1D
from keras.callbacks import EarlyStopping
from keras.layers.normalization import BatchNormalization
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing.text import Tokenizer
from keras.callbacks import Callback, ModelCheckpoint
from keras.regularizers import l2
#from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier,  KerasRegressor
gc.collect()


# In[2]:


#Functions we need - Feature Selector, Fasttext_Estimator, Preprocessing Transformer, Binary_Encoder
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from pandas.api.types import is_numeric_dtype, is_string_dtype
from scipy.sparse.csr import csr_matrix
from sklearn.metrics import mean_squared_error, make_scorer

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

rmse_sklearn = make_scorer(rmse, greater_is_better=False)    


class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, thresh=0, func=np.mean, add_to_orig=False):
        self.cols = cols
        self.thresh = thresh
        self.func = func
        self.add_to_orig = add_to_orig
    
    #@numba.jit        
    def fit(self, X, y):
        self.prior = self.func(y)
        self._dict = {}
        for col in self.cols:
            if isinstance(col, (list, tuple)):
                print('here')
                tmp_df = X.loc[: ,col]
                col = tuple(col)
            else:
                tmp_df = X.loc[: ,[col]]
            tmp_df['y'] = y
            print(tmp_df.columns)
            #tmp_df = pd.DataFrame({'eval_col':X[col].values, 'y':y})
            if isinstance(col, (list, tuple)):
                print('here')
                col = tuple(col)
            self._dict[col] = tmp_df.groupby(col)['y'].apply(lambda x: 
                                self.func(x) if len(x) >= self.thresh  else self.prior).to_dict()
                                
            del tmp_df
        return self
    #@numba.jit
    def transform(self, X, y=None):
        X_transformed = []
        for col in self.cols:
            
            if isinstance(col, (list, tuple)):
                tmp_df = X.loc[:, col]
                enc = tmp_df[col].apply(lambda x: self._dict[tuple(col)][tuple(x)]
                                                                     if tuple(x) in self._dict[tuple(col)]
                                                                     else self.prior, axis=1).values
            else:
                tmp_df = X.loc[:, [col]]
                enc = tmp_df[col].apply(lambda x: self._dict[col][x]
                                                                     if x in self._dict[col]
                                                                     else self.prior).values
            del tmp_df
            X_transformed.append(enc)
        
        X_transformed = np.vstack(X_transformed).T
        
        if self.add_to_orig:
            return np.concatenate((X.values, X_transformed), axis=1)
            
        else:
            return X_transformed

#num_partitions = 30
#num_cores = 16
#from multiprocessing import Pool, cpu_count
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
    #s = re.sub(r"iphone 6+", r"iphonesixplus", s)
    #s = re.sub(r"iphone", r"one", s)
    #s = re.sub(r"2", r"two", s)
    #s = re.sub(r"3", r"three", s)
    #s = re.sub(r"4", r"four", s)
    #s = re.sub(r"5", r"five", s)
    #s = re.sub(r"6", r"six", s)
    #s = re.sub(r"7", r"seven", s)
    #s = re.sub(r"8", r"eight", s)
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
        
    def fit_data(self, data, cols, ngrams=3, max_features=50000):
        c = Counter()
        for col in cols:
            c += Counter(list(chain.from_iterable(data[col].tolist())))
        for i, (w, count) in enumerate(c.most_common(max_features)):
            self.word2index[w] = i + 1
        return
    

            
            
def prepareVocab(name, data, cols, max_features):
    vocab = Vocab_topwords(name)
    vocab.fit_data(data, cols, max_features=max_features)
    
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

def get_cat_1(x): return str(x).lower().split('/')[0]
def get_cat_2(x): return str(x).lower().split('/')[1] if len(str(x).split('/')) > 1 else -1
def get_cat_3(x): return str(x).lower().split('/')[2] if len(str(x).split('/')) > 2 else -1
def get_cat_4(x): return str(x).lower().split('/')[3] if len(str(x).split('/')) > 3 else -1
def get_cat_5(x): return str(x).lower().split('/')[4] if len(str(x).split('/')) > 4 else -1
#def get_cat_3(x): return ' '.join(str(x).lower().split('/')[2:]) if len(str(x).split('/')) > 2 else -1

def applycat1(df): 
    df['cat1'] = df['category_name'].progress_apply(get_cat_1)
    return df

def applycat2(df): 
    df['cat2'] = df['category_name'].progress_apply(get_cat_2)
    return df

def applycat3(df): 
    df['cat3'] = df['category_name'].progress_apply(get_cat_3)
    return df

def applycat4(df): 
    df['cat4'] = df['category_name'].progress_apply(get_cat_4)
    return df

def applycat5(df): 
    df['cat5'] = df['category_name'].progress_apply(get_cat_5)
    return df

def norm3grams(s): return _normalize_and_ngrams(s, 3)

def norm1grams(s): return _normalize_and_ngrams(s, 1)

def applyname(series): return series.progress_apply(norm3grams)

def applycategory(series): return series.progress_apply(norm1grams)

def index2sent1(x, name_vocab): return indexesFromSentence(all_vocab, x, 3, 10)

def name2index(series): return series.progress_apply(lambda x: index2sent1(x, all_vocab))

def applydesc(series):return series.progress_apply(norm1grams)

def index2sent2(x, desc_vocab): return indexesFromSentence(all_vocab, x, 1, 15)

def desc2index(series): return series.progress_apply(lambda x: index2sent2(x, all_vocab))

def isiphonecase(series): return (series.str.contains('iphone', flags=re.IGNORECASE) & 
                                (series.str.contains('case', flags=re.IGNORECASE)) )
def isiphone6(series): return (series.str.contains('iphone', flags=re.IGNORECASE) & 
                        series.str.contains('6|six', flags=re.IGNORECASE) &
                        ~(series.str.contains('plus|\+', flags=re.IGNORECASE)) &
                                ~(series.str.contains('case', flags=re.IGNORECASE)) )

def isiphone6p(series): return (series.str.contains('iphone', flags=re.IGNORECASE) & 
                        series.str.contains('6|six', flags=re.IGNORECASE) &
                        series.str.contains('plus|\+', flags=re.IGNORECASE) &
                                ~(series.str.contains('case', flags=re.IGNORECASE)) )

def isiphone5(series): return (series.str.contains('iphone', flags=re.IGNORECASE) & 
                        series.str.contains('5|five', flags=re.IGNORECASE) &
                        ~(series.str.contains('plus|\+', flags=re.IGNORECASE)) &
                                ~(series.str.contains('case', flags=re.IGNORECASE)) )

def isiphone5p(series): return (series.str.contains('iphone', flags=re.IGNORECASE) & 
                        series.str.contains('5|five', flags=re.IGNORECASE) &
                        series.str.contains('plus|\+', flags=re.IGNORECASE) &
                                ~(series.str.contains('case', flags=re.IGNORECASE)) )

def isiphone7(series): return (series.str.contains('iphone', flags=re.IGNORECASE) & 
                        series.str.contains('7|seven', flags=re.IGNORECASE) &
                        ~(series.str.contains('plus|\+', flags=re.IGNORECASE)) &
                                ~(series.str.contains('case', flags=re.IGNORECASE)) )

def isiphone7p(series): return (series.str.contains('iphone', flags=re.IGNORECASE) & 
                        series.str.contains('7|seven', flags=re.IGNORECASE) &
                        series.str.contains('plus|\+', flags=re.IGNORECASE) &
                                ~(series.str.contains('case', flags=re.IGNORECASE)) )

def read_data(in_path, out_path):
    if False and os.path.exists(os.path.join(out_path, 'train_2.pkl')) and os.path.exists(os.path.join(out_path, 'test_2.pkl')):
        train_data = pd.read_pickle(os.path.join(out_path, 'train_2.pkl'))
        test_data  = pd.read_pickle(os.path.join(out_path, 'test_2.pkl'))
        
        return train_data, test_data
    
    else:
        train_data = pd.read_table(os.path.join(in_path, 'train.tsv'))
        test_data  = pd.read_table(os.path.join(in_path, 'test.tsv'))
    
        train_rows = len(train_data)
        data = pd.concat([train_data, test_data], ignore_index=True)
        
        data['name'] = data['name'].astype(str)
        data['brand_name_cat'] = data['brand_name'].astype(str)
        data['item_description'] = data['item_description'].astype(str)
        #Get name character counts
        #chargrams = CountVectorizer(min_df=10, max_features=30, 
        #                            ngram_range=(1,1), analyzer='char').fit_transform(data['name'])
        #chargrams = pd.DataFrame(chargrams.toarray(), columns=['namechar_'+str(i) for i in range(30)])
        #data = pd.concat([data, chargrams], axis=1)
        data['desc_words'] = data['item_description'].apply(lambda x: len(str(x).split()))
        data['desc_chars'] = data['item_description'].apply(lambda x: len(str(x)))
        data['name_chars'] = data['name'].apply(len)
        data['iphone_case'] = isiphonecase(data['name'])
        data['iphone6'] = isiphone6(data['name'])
        data['iphone6p'] = isiphone6p(data['name'])
        data['iphone5'] = isiphone5(data['name'])
        data['iphone5p'] = isiphone5p(data['name'])
        data['iphone7'] = isiphone7(data['name'])
        data['iphone7p'] = isiphone7p(data['name'])
        data['unlocked_phone'] = data.name.str.contains('unlocked', flags=re.IGNORECASE)
        #ddata = dd.from_pandas(data, 4)
        data['category_words'] = applycategory(data['category_name'].astype(str))
        
        data = applycat1(data)
        data = applycat2(data)
        data = applycat3(data)
        data = applycat4(data)
        data = applycat5(data)
        data.fillna(-1, inplace=True)
        cat_cols = ['category_name', 'brand_name', 'item_condition_id', 'cat1', 'cat2', 'cat3', 'cat4', 'cat5']
        print("Label enoding categoricals")
        for col in cat_cols:
            data[col] = LabelEncoder().fit_transform(data[col].astype(str)).astype(np.int32)
            print(data[col].nunique())
        
        print("Get count features")
        target_enc1 = TargetEncoder(cols=['brand_name'], func=len)
        data['brand_counts'] = target_enc1.fit_transform(data[['brand_name']], data.price)
        data['brand_counts'] = data['brand_counts']/data['brand_counts'].max()

        target_enc2 = TargetEncoder(cols=['category_name'], func=len)
        data['cat_counts'] = target_enc2.fit_transform(data[['category_name']], data.price)
        data['cat_counts'] = data['cat_counts']/data['cat_counts'].max()
        
        target_enc3 = TargetEncoder(cols=['cat1'], func=len)
        data['cat1_counts'] = target_enc3.fit_transform(data[['cat1']], data.price)
        data['cat1_counts'] = data['cat1_counts']/data['cat1_counts'].max()
        
        target_enc4 = TargetEncoder(cols=['cat2'], func=len)
        data['cat2_counts'] = target_enc4.fit_transform(data[['cat2']], data.price)
        data['cat2_counts'] = data['cat2_counts']/data['cat2_counts'].max()
        
        target_enc5 = TargetEncoder(cols=['cat3'], func=len)
        data['cat3_counts'] = target_enc5.fit_transform(data[['cat3']], data.price)
        data['cat3_counts'] = data['cat3_counts']/data['cat3_counts'].max()
        
        print("Tokenizing text columns")
        data['name'] = applyname(data['name'])
        data['item_description'] = applydesc(data['item_description'])
        print("Preparing vocabs")
        #global name_vocab
        #name_vocab = prepareVocab('name', data, ['name'], 50000)
        global all_vocab
        all_vocab = prepareVocab('name', data, ['name', 'category_words', 'brand_name_cat'], 100000)
        
        data['name'] = name2index(data['name'])
        #del name_vocab
        
        print("Transforming text to sequences")
        
        #global desc_vocab
        #desc_vocab = prepareVocab('item_description', data, ['item_description'], 250000)
        data['item_description'] = desc2index(data['item_description'])
        #del desc_vocab
        del data['category_words']
        
        data['name_token_counts'] = data['name'].apply(len)
        data['desc_token_counts'] = data['item_description'].apply(len)
        
        train_data = data.iloc[: train_rows, :]
        train_data = train_data.loc[(train_data.price >= 3) & (train_data.price <= 2100), :].reset_index(drop=True)
        test_data  = data.iloc[train_rows: , :].reset_index(drop=True)
        
        del train_data['test_id']
        del test_data['train_id']
        del data
        gc.collect()
        #print("Writing out new pickles dataframes")
        #train_data.to_pickle(os.path.join(out_path, 'train_2.pkl'))
        #test_data.to_pickle(os.path.join(out_path, 'test_2.pkl'))
        
        return train_data, test_data
#%%
import keras.backend as K
from keras.engine.topology import Layer
class ZeroMaskedEntries(Layer):
    """
    This layer is called after an Embedding layer.
    It zeros out all of the masked-out embeddings.
    It also swallows the mask without passing it on.
    You can change this to default pass-on behavior as follows:

    def compute_mask(self, x, mask=None):
        if not self.mask_zero:
            return None
        else:
            return K.not_equal(x, 0)
    """

    def __init__(self, **kwargs):
        self.support_mask = True
        super(ZeroMaskedEntries, self).__init__(**kwargs)

    def build(self, input_shape):
        self.output_dim = input_shape[1]
        self.repeat_dim = input_shape[2]

    def call(self, x, mask=None):
        #print(mask.shape)
        mask = K.cast(mask, 'float32')
        mask = K.repeat(mask, self.repeat_dim)
        #print(mask.shape)
        mask = K.permute_dimensions(mask, (0, 2, 1))
        return x * mask

    def compute_mask(self, input_shape, input_mask=None):
        return None
    
def mask_aware_mean(x):
    # recreate the masks - all zero rows have been masked
    #mask = K.not_equal(K.sum(K.abs(x), axis=2, keepdims=True), 0)

    # number of that rows are not all zeros
    #n = K.sum(K.cast(mask, 'float32'), axis=1, keepdims=False)
    # compute mask-aware mean of x
    x_mean = K.sum(x, axis=1, keepdims=False)
    #print(x_mean.shape)
    return x_mean

def mask_aware_mean_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 3 
    return (shape[0], shape[2])


# In[6]:

from sklearn.base import BaseEstimator, RegressorMixin
class EM_NNRegressor(BaseEstimator, RegressorMixin):
    
    def __init__(self, embed_cols=None, dense_cols=None, embed_dims=None, 
                 text_embed_cols=None, text_embed_seq_lens=None, 
                 text_embed_dims=None,
                 num_layers=2, multiprocess=False,
                layer_activations=None, layer_dims=None,layer_dropouts=None, epochs=20, batchsize=32,
                optimizer_kwargs=None, val_size=0.1, verbose=1, seed=1, lr=0.001, lr_decay=0.002):
        
        self.embed_cols = embed_cols
        self.dense_cols = dense_cols
        self.embed_dims = embed_dims
        self.text_embed_cols = text_embed_cols
        self.text_embed_dims = text_embed_dims
        #self.text_embed_tokenizers = text_embed_tokenizers
        self.text_embed_seq_lens = text_embed_seq_lens
        self.dense_dims = None
        self.num_layers = num_layers
        self.layer_dims = layer_dims
        self.layer_activations = layer_activations
        self.layer_dropouts = layer_dropouts
        self.epochs = epochs
        self.batchsize = batchsize
        self.optimizer_kwargs = optimizer_kwargs
        self.val_size = val_size
        self.verbose = verbose
        self.multiprocess = multiprocess
        self.seed = seed
        self.lr = lr
        self.lr_decay = lr_decay
        self.model = None
        if self.dense_cols:
            self.dense_dims = len(self.dense_cols)
            
    def splitX(self, X):
        X = X.copy()
        X_splits = []
        
        if self.embed_cols:
            for col in self.embed_cols :
                embed_array = np.asarray(X[col])
                X_splits.append(embed_array)
                
        if self.text_embed_cols:
            for col in self.text_embed_cols:
                embed_array = np.asarray([*X[col].values])
                X_splits.append(embed_array)
                
        if self.dense_cols:
            dense_array = X[self.dense_cols].values.reshape(X.shape[0], -1)
            X_splits.append(dense_array)
        del X
        return X_splits
    
    def get_pretrained_embeddings(self):
        embeddings_index = {}
        f = open(os.path.join('/home/mohsin/Downloads/', 'glove.6B.100d.txt'))
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        
        embedding_matrix = np.zeros(self.text_embed_dims[0])
        for word, i in all_vocab.word2index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        print('Found %s word vectors.' % len(embeddings_index))
        return embedding_matrix
        
    def _build_model(self):
        model_inputs = []
        model_layers = []
        
        if self.embed_cols:
            for col, dim in zip(self.embed_cols, self.embed_dims):
                x1 = Input( shape=(1,), name=col)
                model_inputs.append(x1)
                x1 = Embedding(input_dim=dim[0], output_dim=dim[1],)(x1)
                #x1 = Dropout(0.1)(x1)
                x1 = Reshape(target_shape=(dim[1],))(x1)
                model_layers.append(x1)
                
        if self.text_embed_cols:
            dim, seq_len = self.text_embed_dims[0], self.text_embed_seq_lens[0]
            weights = self.get_pretrained_embeddings()
            embed_layer = Embedding(input_dim=dim[0], output_dim=dim[1],
                                   weights=[weights])
            for col, dim, seq_len in zip(self.text_embed_cols, 
                                                self.text_embed_dims, 
                                                self.text_embed_seq_lens):
                x3 = Input( shape=(seq_len,))
                model_inputs.append(x3)
                x3 = embed_layer(x3)
                #x3 = SimpleRNN(128, return_sequences=True, dropout=0.2)(x3)
                x3 = GRU(64, return_sequences=True)(x3)
                #x3 = GRU(16, return_sequences=True)(x3)
                #x3 = Embedding(input_dim=dim[0], output_dim=dim[1], weights=[weights])(x3)
                
                #x3 = ZeroMaskedEntries()(x3)
                #x3 = Lambda(mask_aware_mean, mask_aware_mean_output_shape)(x3)
                #x3 = Convolution1D(nb_filter=128, filter_length=3,
                #     border_mode='valid', activation='relu',
                #     )(x3)                
                #x3 = MaxPool1D(3)(x3)
                #x3 = Convolution1D(nb_filter=64, filter_length=2,
                #     border_mode='valid', activation='relu',
                #     )(x3)
                #x3 = MaxPool1D(3)(x3)
                #x3 = Flatten()(x3)
                x3 = GlobalAveragePooling1D()(x3)
                print(x3.shape)
                #x3 = Reshape(target_shape=(dim[1],))(x3)
                model_layers.append(x3)
                
        if self.dense_cols:
            x2 = Input( shape=(self.dense_dims, ), name='dense_cols')
            model_inputs.append(x2)
            model_layers.append(x2)
        
        del x1, x2, x3
        print(model_layers)
        x = concatenate(model_layers)
        print(x.shape)
        if self.num_layers > 0:
            for dim, drops in zip(self.layer_dims, self.layer_dropouts):
                x = BatchNormalization()(x)
                x = Dropout(rate=drops, seed=self.seed)(x)
                x = Dense(dim, kernel_initializer='he_normal')(x)
                #x = Dense(dim, activation='tanh', kernel_initializer='he_normal')(x)
                x = LeakyReLU()(x)
        
        x = BatchNormalization()(x)
        x = Dropout(0.02, seed=self.seed)(x)
        output = Dense(1, activation='linear', kernel_initializer='he_normal')(x)
        print(output.shape)
        model = Model(inputs=model_inputs, outputs=output)
        #print(model.summary())
        adam = RMSprop(lr=self.lr, decay=self.lr_decay)
        #adam = Nadam(lr=0.0015)
        model.compile(optimizer=adam, loss='mean_squared_error')
        
        return model 
    
    
    def fit(self, X, y):
        self.model = self._build_model()
        if self.val_size > 0:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.val_size, random_state=self.seed)
            print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)
            X_train_splits = self.splitX(X_train)
            X_val_splits = self.splitX(X_val)
            #callbacks= [ModelCheckpoint("embed_NN_"+str(self.seed)+".check", save_best_only=True, verbose=1)]
            if self.multiprocess == False:
                self.model.fit(X_train_splits, y_train, batch_size=self.batchsize, epochs=self.epochs,
                               verbose=self.verbose,
                              validation_data=(X_val_splits, y_val), shuffle=True,
                              #callbacks=callbacks
                              )
                del X_train, X_val, y_train, y_val,  X_train_splits, X_val_splits
                gc.collect()
            else:
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.val_size, random_state=1)
                del X_train, X_val, y_train, y_val,
        else:
            print(X.shape, y.shape)
            X_splits = self.splitX(X)
            self.model.fit(X_splits, y, batch_size=self.batchsize, epochs=self.epochs,
               verbose=self.verbose, shuffle=True,)
            

        
        return self
    
    def predict(self, X, y=None):
        
        if self.model:
            #self.model = load_model("embed_NN_"+str(self.seed)+".check")
            X_splits = self.splitX(X)
            y_hat = self.model.predict(X_splits)
        else:
            raise ValueError("Model not fit yet")
            
        return y_hat


# In[7]:
if __name__ == "__main__":
    print("Reading and preprocessing data")
    train_data, test_data = read_data("../input", "./")

    print(train_data.columns)
    
    #optimized on 5 fold cv 
    params = [55, 60, 7, 10, 17, 45, 252, 0.2, 0.010774494286264107, 0.00062998520310506793]
    brand_dim, category_dim, condition_dim, cat1_dim, cat2_dim, cat3_dim, dense_dim, dense_drop, lr, lr_decay = params
    
    print("Initalizing neural network")
    nnet2 = EM_NNRegressor(embed_cols=['brand_name','category_name','item_condition_id', 'cat1', 'cat2', 'cat3', 'cat4', 'cat5'], 
                      embed_dims=[(6000, brand_dim),
                                  (1500, category_dim), 
                                  (5, condition_dim), 
                                  (15,cat1_dim), 
                                  (120, cat2_dim), 
                                  (900, cat3_dim), (7,4), (3,2)],
                      text_embed_cols=['name', 'item_description'],
                      text_embed_dims=[(100001, 100), (100001, 100)],
                      text_embed_seq_lens =[10, 15], 
                      #text_embed_cols=['name'],
                      #text_embed_dims=[(50000, 80)],
                      #text_embed_seq_lens =[10], 
                      dense_cols=['shipping', 'desc_words', 'desc_chars', 'name_chars',
                                'iphone_case', 'iphone6', 'iphone6p',
                                'iphone5', 'iphone5p', 'iphone7', 'iphone7p', 'unlocked_phone',
                                  'brand_counts', 'cat_counts',
                                   'cat1_counts', 'cat2_counts', 'cat3_counts',
                                  'name_token_counts','desc_token_counts'
                                 ],
                      epochs=6,
                      batchsize=2048,
                      num_layers = 2,
                      layer_dropouts=[dense_drop],
                      layer_dims=[dense_dim],
                      val_size=0,
                      seed=2,
                      lr=lr,
                      lr_decay=lr_decay,
                     )
    #print(nnet2.model.summary())
    preds = cross_val_predict(nnet2, train_data, np.log1p(train_data.price), cv=5, verbose=1, n_jobs=1, random_state=1)
    gc.collect()
        #score = np.mean(scores)
    score = rmse(np.log1p(train_data.price), preds)
    print("CV score", score)    
    print("Fitting model")
    #%%    
    print("Fitting model")
    nnet2.fit(train_data, np.log1p(train_data.price))
    
    print("Predict on test data")
    test_preds = nnet2.predict(test_data)
    
    print("Write out submission")
    submission: pd.DataFrame = test_data[['test_id']]
    submission['price'] = np.expm1(test_preds)
    submission.price = submission.price.clip(3, 2000)
    submission.to_csv("embedding_nn_v2.csv", index=False)