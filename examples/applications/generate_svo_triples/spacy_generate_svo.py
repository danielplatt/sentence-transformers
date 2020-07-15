# from nltk.corpus import wordnet as wn
#
# all_nouns = set()
#
# for synset in list(wn.all_synsets('n')):
#     for noun in synset.lemma_names():
#         all_nouns.add(noun.replace("_", " "))
#
# all_nouns_lowercase = [noun.lower() for noun in all_nouns]

import enchant
broker = enchant.Broker()
broker.describe()
broker.list_languages()

exit()

pronouns = {'all','another','any','anybody','anyone','anything','as','aught','both','each','each other','either','enough','everybody','everyone','everything','few','he','her','hers','herself','him','himself','his','I','idem','it','its','itself','many','me','mine','most','my','myself','naught','neither','no one','nobody','none','nothing','nought','one','one another','other','others','ought','our','ours','ourself','ourselves','several','she','some','somebody','someone','something','somewhat','such','suchlike','that','thee','their','theirs','theirself','theirselves','them','themself','themselves','there','these','they','thine','this','those','thou','thy','thyself','us','we','what','whatever','whatnot','whatsoever','whence','where','whereby','wherefrom','wherein','whereinto','whereof','whereon','wherever','wheresoever','whereto','whereunto','wherewith','wherewithal','whether','which','whichever','whichsoever','who','whoever','whom','whomever','whomso','whomsoever','whose','whosever','whosesoever','whoso','whosoever','ye','yon','yonder','you','your','yours','yourself','yourselves'}

import spacy
nlp = spacy.load('en_core_web_sm')

import json
import codecs

with codecs.open('news_0056951.json', 'r', 'utf-8-sig') as json_file:
    news_item = json.load(json_file)
    news_item_text = news_item['text']
    print(news_item_text)

docc = nlp(news_item_text)




import textacy
import re
triples = textacy.extract.subject_verb_object_triples(docc)

for trip in triples:
    good_triple = True
    for k in [0,1,2]:
        # print(trip[0])
        # print(trip[k].lemma_)
        number_of_capital_letters = sum(1 for c in trip[k].text if c.isupper())
        good_triple = good_triple and all(x.isalpha() for x in trip[k].text) and trip[k].text not in pronouns and number_of_capital_letters <=1 and trip[k].text != 'to' and trip[k].text != 'be'
    if good_triple:
        print(trip)

    # print(trip)
    # if all(x.isalpha() for x in trip[1].text) and trip[0].text.lower() in all_nouns_lowercase and trip[1].text.lower() in all_nouns_lowercase:
    #     print(trip)




