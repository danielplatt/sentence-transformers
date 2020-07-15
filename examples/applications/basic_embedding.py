"""
This basic example loads a pre-trained model from the web and uses it to
generate sentence embeddings for a given list of sentences.
"""

from sentence_transformers import SentenceTransformer, LoggingHandler
import numpy as np
import logging

#### Just some code to print debug information to stdout
np.set_printoptions(threshold=100)

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout



# Load Sentence model (based on BERT) from URL
model = SentenceTransformer('bert-base-nli-mean-tokens')

# Embed a list of sentences
# positive_sentences = ['She is nice.','It is hot.','Today is Monday.','Today is Tuesday.','Today is Wednesday.','School is boring.','A TV is an electronic device.','All cats are brown.']
# negative_sentences = ['She is not nice.','It is not hot.','Today is not Monday.','Today is not Tuesday.','Today is not Wednesday.','School is not boring.','A TV is not an electronic device.','Not all cats are brown.']

#################################

positive_sentences = ['He chases her.', 'The low interest rates caused the financial crisis.', 'The author blamed the cat.', 'The dog barks at him.', 'The crowd intimidates the bird.', 'Because it was late, Max walked Angela home.', 'The opportunity created a problem.', 'He chases her.']

negative_sentences = ['She chases him.', 'The financial crisis caused the low interest rates.', 'The cat blamed the author.', 'He barks at the dog.', 'The bird intimidates the crowd.', 'Because it was late, Angela walked Max home.', 'A problem created the opportunity.', 'He runs after her.']

#################################

import json

# with open('active_sentences.json') as json_file:
#     positive_sentences = json.load(json_file)
# with open('passive_sentences.json') as json_file:
#     negative_sentences = json.load(json_file)

import json
import codecs

with codecs.open('positive_sentences.json', 'r', 'utf-8-sig') as json_file:
    positive_sentences = json.load(json_file)
with codecs.open('negative_sentences.json', 'r', 'utf-8-sig') as json_file:
    negative_sentences = json.load(json_file)

positive_sentence_embeddings = model.encode(positive_sentences)
negative_sentence_embeddings = model.encode(negative_sentences)

import matplotlib
import matplotlib.pyplot as plt

all_pos_embeddings = []
all_neg_embeddings = []
mixed_embeddings = []
differences = []
for pos_sentence, pos_embedding, neg_sentence, neg_embedding in zip(positive_sentences,positive_sentence_embeddings,negative_sentences,negative_sentence_embeddings):
    # print("Sentence:", sentence)
    # print("Embedding:", embedding)
    # print("")
    all_pos_embeddings += [pos_embedding]
    all_neg_embeddings += [neg_embedding]
    mixed_embeddings += [pos_embedding, neg_embedding]
    differences += [[pos-neg for pos, neg in zip(pos_embedding, neg_embedding)]]


separator = [0 for i in range(768)]

display_embeddings = all_pos_embeddings + [separator for i in range(20)] +all_neg_embeddings + [separator for i in range(20)] + mixed_embeddings + [separator for i in range(20)] + differences

fig, ax = plt.subplots(figsize=(200, 10))
ax.imshow(display_embeddings, cmap='RdYlBu', aspect='auto')

plt.tight_layout()
plt.savefig('sentence_embeddings')
plt.show()

from scipy import spatial
def cosine_dist(x,y):
    return 1-spatial.distance.cosine(x,y)

cosine_distance_matrix = [[cosine_dist(pos,neg+diff) for diff in differences] for pos,neg in zip(positive_sentence_embeddings,negative_sentence_embeddings)]

print(cosine_distance_matrix)

print(np.array(cosine_distance_matrix).min())
print(np.array(cosine_distance_matrix).max())

fig, ax = plt.subplots(figsize=(40, 40))
ax.imshow(cosine_distance_matrix, cmap='RdYlBu', aspect='auto')

plt.tight_layout()
plt.savefig('cosine_distance_matrix')
plt.show()
