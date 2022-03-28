from transformers import *
from sklearn.datasets import fetch_20newsgroups

src = "en"
dst = "ru"
task_name = f"translation_{src}_to_{dst}"
model_name = f"Helsinki-NLP/opus-mt-{src}-{dst}"
translator = pipeline(task_name, model=model_name, tokenizer=model_name)

categories = ['sci.space']
remove = ['headers', 'footers', 'quotes']
text = fetch_20newsgroups(subset='test', remove=remove)

count, i = 0, 0
alp_lat_cap = set(chr(i) for i in range(ord('A'), ord('Z') + 1))
alp_lat = set(chr(i) for i in range(ord('a'), ord('z') + 1))
alp_lat.update(alp_lat_cap)
alp_lat.add('!')
alp_lat.add('?')
alp_lat.add('-')
is_lat = False

while (i < 100):
    cur_sentens = text.data[count].split('.')
    for el in cur_sentens:
        if len(el) < 6:
            count += 1
            continue
        tr = translator(el)[0]["translation_text"]
        for sp in tr:
            if sp in alp_lat:
                is_lat = True
        if len(tr) < 6 or tr[0].isdigit() or tr[0] == ',' or is_lat or \
                tr[0] == '(' or tr[0] == '"' or tr[0] == ')' or tr[0] == ':' or \
                tr[0] == '[' or tr.isupper() or tr.islower() or tr[0] == "«":
            count += 1
            is_lat = False
            continue
        # if tr[-1] == ')' or tr[-1] == '"':
        #     tr = tr[:-1]
        if tr[-1] == '.':
            print(i + 1, tr, '\n')
        else:
            print(i + 1, tr + '.', '\n')
        if i == 99:
            i += 1
            break
        count += 1
        i += 1
