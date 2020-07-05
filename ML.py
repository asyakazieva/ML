# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 16:30:02 2019

"""

import pandas as pd
import re
import gensim
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(gensim.parsing.preprocessing.remove_stopwords(str(sentence)), deacc = True))  # deacc=True removes punctuations

def dfs_to_list(dfs_dict):
    for k, v in dfs_dict.items(): yield(str(dictionary.id2token[k]) + ';' + str(v))
    
df = pd.read_excel(r'c:\testdata\requests.xlsx', dtype={'DESCRIPTION':str,'ROLE_SCOPE_TEXT':str,'SOL_PROFILE_DESC':str},keep_default_na=False)
 
 #Filter only English language
is_english = df['Lang.detect.']=='en'
df = df[is_english]
 #Filter out short texts
 
#concatenate descirption + scope of tasks into one column
#df['REQUEST_TEXT'] = df['DESCRIPTION'] + ' ' + df['ROLE_SCOPE_TEXT']
print('prepare word list...')
data = df['ROLE_SCOPE_TEXT'].values.tolist()
print('END prepare word list...')
data_words = list(sent_to_words(data))
#we want to leave only docs > 3 words
data_words = [ doc for doc in data_words if len(doc) > 3 ]

print('creating dictionary')
dictionary = gensim.corpora.Dictionary(data_words)


corpus = [dictionary.doc2bow(text) for text in data_words]
tfidf = gensim.models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
lsi = gensim.models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=50)
lsi.print_topics(10)
