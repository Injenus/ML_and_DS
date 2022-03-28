from transformers import *
from sklearn.datasets import fetch_20newsgroups

src = "en"
dst = "ru"
task_name = f"translation_{src}_to_{dst}"
model_name = f"Helsinki-NLP/opus-mt-{src}-{dst}"
translator = pipeline(task_name, model=model_name, tokenizer=model_name)

categories = ['sci.space']
remove = ['headers', 'footers', 'quotes']
text = fetch_20newsgroups(subset='test', categories=categories, remove=remove)

count = 0
while (count < 100):
    cur_sentens = text.data[count].split('.')
    for el in cur_sentens:
        tr = translator(el)[0]["translation_text"]
        if len(tr) < 2:
            continue
        if tr[0] == '(' or tr[0] == '"':
            tr = tr[1:]
        # if tr[-1] == ')' or tr[-1] == '"':
        #     tr = tr[:-1]
        if tr[-1] == '.' or tr[-1] == '!' or tr[-1] == '?':
            print(tr, count + 1, '\n')
        else:
            print(tr + '.', count + 1, '\n')
        count += 1
