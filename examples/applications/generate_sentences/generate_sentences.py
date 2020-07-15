import lemminflect as lem

import json
import codecs

with codecs.open('svo_triples_lemmatised.txt', 'r', 'utf-8-sig') as json_file:
    svo_triples = json.load(json_file)

svo_triples = svo_triples[3:]

pos_sentences = []
neg_sentences = []
passive_sentences = []
swapped_sentences = []

for triple in svo_triples:
    triple_subject = lem.getInflection(triple[0], tag='NNS', inflect_oov=False)
    triple_verb = lem.getInflection(triple[1], tag='VBD', inflect_oov=False)
    triple_negative_verb = lem.getInflection(triple[1], tag='VB', inflect_oov=False)
    triple_passive_verb = lem.getInflection(triple[1], tag='VBN', inflect_oov=False)
    triple_object = lem.getInflection(triple[2], tag='NNS', inflect_oov=False)
    if triple_subject != () and triple_verb != () and triple_object != ():
        #print((triple_subject,triple_verb,triple_object))
        #print(triple)
        pos_sentences += [triple_subject[0].capitalize() + " " + triple_verb[0] + " " + triple_object[0] + "."]
        neg_sentences += [triple_subject[0].capitalize() + " did not " + triple_negative_verb[0] + " " + triple_object[0] + "."]
        passive_sentences += [triple_object[0].capitalize() + " were " + triple_passive_verb[0] + " by " + triple_subject[0] + "."]
        swapped_sentences += [triple_object[0].capitalize() + " " + triple_verb[0] + " " + triple_subject[0] + "."]


print(pos_sentences)
print(neg_sentences)
print(passive_sentences)
print(swapped_sentences)

with open('svo_generated_sentences.txt', 'w') as outfile:
    json.dump({'pos_sentences': pos_sentences, 'neg_sentences': neg_sentences, 'passive_sentences': passive_sentences, 'swapped_sentences': swapped_sentences}, outfile)