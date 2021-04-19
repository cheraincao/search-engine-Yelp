## Unsupervised summarization model -- static summary

from gensim.summarization import summarize
from gensim.models import Word2Vec
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')
import skipthoughts
from skipthoughts import load_model, Encoder
import os
import theano
import theano.tensor as tensor
import pickle as pkl
import numpy as np
import copy
from collections import OrderedDict, defaultdict
from scipy.linalg import norm
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import pandas as pd

## load dataset
df1 = pd.read_json('df1.json')

def tokenize_text(text):
    """
    Splits the text into individual sentences
    """
    sentences = nltk.sent_tokenize(text)
    return sentences


def skipthought_encode(sentences):
    """
    Obtains sentence embeddings for each sentence in the text
    """
    # print('Loading pre-trained models...')
    model = skipthoughts.load_model()
    encoder = skipthoughts.Encoder(model)
    # print('Encoding sentences...')
    encoded = encoder.encode(sentences)
    return encoded


def summarize(text):
    """
    Performs summarization of text
    """
    summary = []
    # print('Splitting into sentences...')
    token_text = tokenize_text(text)
    # print('Starting to encode...')
    enc_text = skipthought_encode(token_text)
    # print('Encoding Finished')
    n_clusters = int(np.ceil(len(enc_text) * 0.07)) # n_clusters is related to the length of summary
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans = kmeans.fit(enc_text)
    avg = []
    closest = []
    for j in range(n_clusters):
        idx = np.where(kmeans.labels_ == j)[0]
        avg.append(np.mean(idx))
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, \
                                               enc_text)
    ordering = sorted(range(n_clusters), key=lambda k: avg[k])
    summary = ' '.join([token_text[closest[idx]] for idx in ordering])
    # print('Clustering Finished')
    return summary


# def extract_summary(id):
#     """
#     Input business_id, get corresponding summary
#     """
#     text = str(business_tip[business_tip['business_id']==id]['tip_text'].values)
#     tip_summary = summarize(text)
#     return tip_summary


def get_summary(df):
    """
    Input dataframe, add new feature 'summary'
    """
    for i in range(len(df)):
        text = df.at[i,'tip_text']
        if len(text) > 0:
            post_text = summarize(text)
        else:
            post_text = ''
        df.at[i,'summary'] = post_text
    return df


data1 = get_summary(df1)
raw1 = data1[['business_id','summary']]
raw1.to_json('data1.json')
print('Finish saving')