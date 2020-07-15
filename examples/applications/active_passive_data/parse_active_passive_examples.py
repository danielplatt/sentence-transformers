import pandas as pd
df = pd.read_csv('rawfile.csv', engine='python', header=None)
# print(df)

sentences = df.values.tolist()

# print(sentences)

sentences = [(sent[0].split(" ("))[0] for sent in sentences]

# print(sentences)

act_sent = []
pass_sent = []

for i in range(int(len(sentences)/2)):
    act_sent += [sentences[2*i]]
    # print(sentences[2*i])
    # print(act_sent)
    pass_sent += [sentences[2 * i+1]]

print(act_sent)
print(pass_sent)

import json
with open('../active_sentences.json', 'w') as f:
    json.dump(act_sent, f)
with open('../passive_sentences.json', 'w') as f:
    json.dump(pass_sent, f)