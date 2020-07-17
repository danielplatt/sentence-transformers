"""
This basic example loads a pre-trained model from the web and uses it to
generate sentence embeddings for a given list of sentences.
"""

from sentence_transformers import SentenceTransformer, LoggingHandler
import numpy as np
import logging
import scipy

import matplotlib
import matplotlib.pyplot as plt

#### Just some code to print debug information to stdout
np.set_printoptions(threshold=100)

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

import json

def load_active_passive_dataset():
    with open('datasets/self_generated/svo_generated_sentences.json') as json_file:
        all_sentences = json.load(json_file)
        normal_sentences = all_sentences['pos_sentences']
        passive_sentences = all_sentences['passive_sentences']
        return "active_passive_dataset", normal_sentences, passive_sentences

def load_google_PAWS():
    import pandas as pd
    df = pd.read_csv('google_paws/train.tsv', engine='python', sep='\t')
    sentences = df.values.tolist()
    sentences_A = []
    sentences_B = []
    for id, sentence_A, sentence_B, label in sentences:
        if label == 1:
            sentences_A += [sentence_A]
            sentences_B += [sentence_B]
    return "google_paws", sentences_A, sentences_B

def load_positive_negative_dataset():
    with open('datasets/self_generated/svo_generated_sentences.json') as json_file:
        all_sentences = json.load(json_file)
        normal_sentences = all_sentences['pos_sentences']
        negative_sentences = all_sentences['neg_sentences']
        return "positive_negative_dataset", normal_sentences, negative_sentences

def load_female_male_dataset():
    female_words = ["she", "her", "woman", "herself", "daughter", "mother", "gal", "girl", "female"]
    male_words = ["he", "his", "man", "himself", "son", "father", "guy", "boy", "male"]
    return "female_male_dataset", female_words, male_words

def give_difference_vectors(list_of_vectors_A, list_of_vectors_B):
    assert(len(list_of_vectors_A)==len(list_of_vectors_B))
    list_diff = lambda list1, list2: [item1 - item2 for item1, item2 in zip(list1, list2)]
    differences = [list_diff(female_vec, male_vec) for female_vec, male_vec in zip(list_of_vectors_A, list_of_vectors_B)]
    return differences

def distance(subspace, linear_combination_factors, point):
    assert(len(linear_combination_factors) == len(subspace))
    linear_combination_factors = np.array(linear_combination_factors)
    subspace = np.array(subspace)

    # next two lines can be written as a matrix multiplication...
    linear_combination = np.array([factor*basis_vector for factor, basis_vector in zip(linear_combination_factors, subspace)])
    linear_combination_vector = np.sum(linear_combination, axis=0)

    point = np.array(point)
    return np.linalg.norm(linear_combination_vector - point)

# print(distance([[0,1]],[100],[1,0]))

def find_nearest_point_in_subspace(subspace, point):
    function = lambda linear_factors : distance(subspace, linear_factors, point)
    initial_guess = [0 for k in range(len(subspace))]
    minimiser = scipy.optimize.minimize(function, initial_guess)

    # next two lines can be written as a matrix multiplication...
    subspace = np.array(subspace)
    linear_combination = np.array([factor * basis_vector for factor, basis_vector in zip(minimiser.x, subspace)])
    linear_combination_vector = np.sum(linear_combination, axis=0)

    return linear_combination_vector

print(find_nearest_point_in_subspace([[0,1]],[1,1]))

def cosine_dist(x,y):
    return 1-scipy.spatial.distance.cosine(x,y)

def vector_subspace_angle(subspace, vector):
    return cosine_dist(find_nearest_point_in_subspace(subspace,vector), vector)

def vector_subspace_distance(subspace, vector):
    return np.linalg.norm(find_nearest_point_in_subspace(subspace,vector)-vector)

print("vector_subspace_angle is invariant under scaling: ")
print(vector_subspace_angle([[0,1]],[1,1]),vector_subspace_angle([[0,1]],[5,5]))
print("vector_subspace_distance is linear under scaling: ")
print(vector_subspace_distance([[0,1]],[1,1]),vector_subspace_distance([[0,1]],[5,5]))

exit()

def store_principal_components(name, components):
    try:
        with open('principal_components.json') as json_file:
            all_component_data = json.load(json_file)

    except FileNotFoundError as e:
        print("File principal_components.json does not yet exist. Creating it.")
        all_component_data = {}

    except json.decoder.JSONDecodeError as e:
        print("File principal_components.json appears to be empty. All contents of the file will be overwritten.")
        all_component_data = {}

    all_component_data[name] = components

    with open('principal_components.json', 'w') as outfile:
        json.dump(all_component_data, outfile)

# Load Sentence model (based on BERT) from URL
model = SentenceTransformer('bert-base-nli-mean-tokens')


# dataset_name, sentences_A, sentences_B = load_active_passive_dataset()
dataset_name, sentences_A, sentences_B = load_google_PAWS()
# dataset_name, sentences_A, sentences_B = load_female_male_dataset()

embeddings_A = model.encode(sentences_A)
embeddings_B = model.encode(sentences_B)

import random
embeddings_C = embeddings_B.copy()
random.shuffle(embeddings_C)

differences_AB = give_difference_vectors(embeddings_A, embeddings_B)
differences_AC = give_difference_vectors(embeddings_A, embeddings_C)

from sklearn.decomposition import PCA

pca_analyser_AB = PCA()#(n_components=5)
pca_analyser_AB.fit(differences_AB)

pca_analyser_AC = PCA()#(n_components=5)
pca_analyser_AC.fit(differences_AC)

print((pca_analyser_AB.explained_variance_,pca_analyser_AB.explained_variance_ratio_))
print((pca_analyser_AC.explained_variance_,pca_analyser_AC.explained_variance_ratio_))

print(pca_analyser_AB.components_)
print(pca_analyser_AC.components_)

store_principal_components(dataset_name, {'components_': pca_analyser_AB.components_.tolist(), 'explained_variance_': pca_analyser_AB.explained_variance_.tolist(), 'explained_variance_ratio_': pca_analyser_AB.explained_variance_.tolist()})
store_principal_components(dataset_name, {'components_': pca_analyser_AC.components_.tolist(), 'explained_variance_': pca_analyser_AC.explained_variance_ratio_.tolist(), 'explained_variance_ratio_': pca_analyser_AC.explained_variance_ratio_.tolist()})

exit()

# all_pos_embeddings = []
# all_neg_embeddings = []
# mixed_embeddings = []
# differences = []
# for pos_sentence, pos_embedding, neg_sentence, neg_embedding in zip(positive_sentences,positive_sentence_embeddings,negative_sentences,negative_sentence_embeddings):
#     # print("Sentence:", sentence)
#     # print("Embedding:", embedding)
#     # print("")
#     all_pos_embeddings += [pos_embedding]
#     all_neg_embeddings += [neg_embedding]
#     mixed_embeddings += [pos_embedding, neg_embedding]
#     differences += [[pos-neg for pos, neg in zip(pos_embedding, neg_embedding)]]
#
#
# separator = [0 for i in range(768)]
#
# display_embeddings = all_pos_embeddings + [separator for i in range(20)] +all_neg_embeddings + [separator for i in range(20)] + mixed_embeddings + [separator for i in range(20)] + differences
#
# fig, ax = plt.subplots(figsize=(200, 10))
# ax.imshow(display_embeddings, cmap='RdYlBu', aspect='auto')
#
# plt.tight_layout()
# plt.savefig('sentence_embeddings')
# plt.show()
#
#
#
#
# cosine_distance_matrix = [[cosine_dist(pos,neg+diff) for diff in differences] for pos,neg in zip(positive_sentence_embeddings,negative_sentence_embeddings)]
#
# print(cosine_distance_matrix)
#
# print(np.array(cosine_distance_matrix).min())
# print(np.array(cosine_distance_matrix).max())
#
# fig, ax = plt.subplots(figsize=(40, 40))
# ax.imshow(cosine_distance_matrix, cmap='RdYlBu', aspect='auto')
#
# plt.tight_layout()
# plt.savefig('cosine_distance_matrix')
# plt.show()
