%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
# Sat, 13 Jun 2020 13:25:16
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
# Sat, 13 Jun 2020 13:25:17
from static_grader import grader# Sat, 13 Jun 2020 13:25:23
import gzip
import ujson as json

with gzip.open("data/amazon_electronics_reviews_training.json.gz", "r") as f:                                  
    data = [json.loads(line) for line in f]# Sat, 13 Jun 2020 13:25:44
%%bash
mkdir data
wget http://dataincubator-wqu.s3.amazonaws.com/mldata/amazon_electronics_reviews_training.json.gz -nc -P ./data# Sat, 13 Jun 2020 13:25:54
import gzip
import ujson as json

with gzip.open("data/amazon_electronics_reviews_training.json.gz", "r") as f:                                  
    data = [json.loads(line) for line in f]%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
# Sat, 13 Jun 2020 13:34:02
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
# Sat, 13 Jun 2020 13:34:05
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
# Sat, 13 Jun 2020 13:34:05
from static_grader import grader# Sat, 13 Jun 2020 13:34:05
%%bash
mkdir data
wget http://dataincubator-wqu.s3.amazonaws.com/mldata/amazon_electronics_reviews_training.json.gz -nc -P ./data# Sat, 13 Jun 2020 13:34:05
import gzip
import ujson as json

with gzip.open("data/amazon_electronics_reviews_training.json.gz", "r") as f:                                  
    data = [json.loads(line) for line in f]# Sat, 13 Jun 2020 13:34:06
data[0]
# Sat, 13 Jun 2020 13:34:06
X=data[0]
X['overall']# Sat, 13 Jun 2020 13:34:06

ratings = [X['overall'] for X in data]# Sat, 13 Jun 2020 13:34:06
ratings[:10]# Sat, 13 Jun 2020 13:34:06
from sklearn.base import BaseEstimator, TransformerMixin

class KeySelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    
    def fit(self, X, Y=None):
        return self
    
    def transform(self, X):
        return[d[self.key] for d in X]# Sat, 13 Jun 2020 13:34:06
ks = KeySelector('reviewText')
x_trans = ks.fit_transform(data)
print(x_trans[0])
# Sat, 13 Jun 2020 13:34:06
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import Ridge# Sat, 13 Jun 2020 13:34:06

bag_of_words_model = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', HashingVectorizer()),
    ('regressor', Ridge(alpha=0.6))
])

bag_of_words_model.fit(data, ratings);# Sat, 13 Jun 2020 13:34:18
grader.score.nlp__bag_of_words_model(bag_of_words_model.predict)# Sat, 13 Jun 2020 13:34:19
from sklearn.linear_model import Lasso# Sat, 13 Jun 2020 13:34:19
normalized_model = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', TfidfVectorizer()),
    ('predictor', Ridge())
])# Sat, 13 Jun 2020 13:34:19

normalized_model.fit(data, ratings);# Sat, 13 Jun 2020 13:34:30
grader.score.nlp__normalized_model(normalized_model.predict)# Sat, 13 Jun 2020 13:34:31

bigrams_model = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', HashingVectorizer(ngram_range=(1,2))),
    ('predictor', Ridge(alpha=0.6))
])# Sat, 13 Jun 2020 13:34:31
bigrams_model.fit(data, ratings);# Sat, 13 Jun 2020 13:35:00
grader.score.nlp__bigrams_model(bigrams_model.predict)# Sat, 13 Jun 2020 13:35:01
%%bash
wget http://dataincubator-wqu.s3.amazonaws.com/mldata/amazon_one_and_five_star_reviews.json.gz -nc -P ./data# Sat, 13 Jun 2020 13:35:01
del data, ratings# Sat, 13 Jun 2020 13:35:02
import numpy as np
from sklearn.naive_bayes import MultinomialNB

with gzip.open("data/amazon_one_and_five_star_reviews.json.gz", "r") as f:
    data_polarity = [json.loads(line) for line in f]

ratings = [row['overall'] for row in data_polarity]
# Sat, 13 Jun 2020 13:35:02
pipe = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', TfidfVectorizer(stop_words='english')),
    ('predictor', MultinomialNB())
])

pipe.fit(data_polarity, ratings);# Sat, 13 Jun 2020 13:35:03
#get features (vocab) from model
feat_to_token = pipe['vectorizer'].get_feature_names()

#get the log probability from model
log_prob = pipe['predictor'].feature_log_prob_

#collapse log probability into one row
polarity = log_prob[0, :] - log_prob[1, :]

#combine polarity and feature names
most_polar = sorted(list(zip(polarity, feat_to_token)))# Sat, 13 Jun 2020 13:35:03

n=25
most_polar = most_polar[:n] + most_polar[-n:]# Sat, 13 Jun 2020 13:35:03

top_50 = [term for score, term in most_polar]# Sat, 13 Jun 2020 13:35:03
grader.score.nlp__most_polar(top_50)# Sat, 13 Jun 2020 13:35:03
from sklearn.decomposition import NMF
 # Sat, 13 Jun 2020 13:35:16
grader.score.nlp__most_polar(top_50)# Sat, 13 Jun 2020 13:36:25
grader.score.nlp__most_polar(top_50)# Sat, 13 Jun 2020 13:36:29
from sklearn.decomposition import NMF
 