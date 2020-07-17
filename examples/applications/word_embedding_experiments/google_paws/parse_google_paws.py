import pandas as pd
df = pd.read_csv('train.tsv', engine='python', sep='\t')
# print(df)

sentences = df.values.tolist()

# print(sentences[10])

sentences_A = []
sentences_B = []

for id, sentence_A, sentence_B, label in sentences:
    if label == 1:
        sentences_A += [sentence_A]
        sentences_B += [sentence_B]
        print("X")

print(sentences_A[50])
print(sentences_B[50])