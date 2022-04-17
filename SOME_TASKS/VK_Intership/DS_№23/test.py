import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import datasets
from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize

categories = ['sci.space']
remove = ['headers', 'footers', 'quotes']
text = fetch_20newsgroups(subset='test', categories=categories, remove=remove)
df_text = pd.DataFrame(text.data)
print(df_text)

print(text.filenames.shape)

cur_sentens = text.data[0].split('.')
print(cur_sentens)

alp_lat = set(chr(i) for i in range (ord('a'),ord('z')+1))
w={1,2}
w.update(alp_lat)
print(w,alp_lat)

a = "("
print(a.isupper())
if not a.isupper():
    print(a.upper())