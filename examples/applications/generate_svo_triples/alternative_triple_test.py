import spacy
from textpipeliner import PipelineEngine, Context
from textpipeliner.pipes import *

import json
import codecs

with codecs.open('news_0056951.json', 'r', 'utf-8-sig') as json_file:
    news_item = json.load(json_file)
    news_item_text = news_item['text']
    print(news_item_text)



nlp = spacy.load("en_core_web_sm")
doc = nlp(news_item_text)
# doc = nlp(
#     "The Empire of Japan aimed to dominate Asia and the Pacific and was "
#     "already at war with the Republic of China in 1937, but the world war is "
#     "generally said to have begun on 1 September 1939 with the invasion of "
#     "Poland by Germany and subsequent declarations of war on Germany by "
#     "France and the United Kingdom. From late 1939 to early 1941, in a "
#     "series of campaigns and treaties, Germany conquered or controlled much "
#     "of continental Europe, and formed the Axis alliance with Italy and "
#     "Japan. Under the Molotov-Ribbentrop Pact of August 1939, Germany and the "
#     "Soviet Union partitioned and annexed territories of their European "
#     "neighbours, Poland, Finland, Romania and the Baltic states. The war "
#     "continued primarily between the European Axis powers and the coalition "
#     "of the United Kingdom and the British Commonwealth, with campaigns "
#     "including the North Africa and East Africa campaigns, the aerial Battle "
#     "of Britain, the Blitz bombing campaign, the Balkan Campaign as well as "
#     "the long-running Battle of the Atlantic. In June 1941, the European Axis "
#     "powers launched an invasion of the Soviet Union, opening the largest "
#     "land theatre of war in history, which trapped the major part of the "
#     "Axis' military forces into a war of attrition. In December 1941, Japan "
#     "attacked the United States and European territories in the Pacific "
#     "Ocean, and quickly conquered much of the Western Pacific.")

pipes_structure = [
    SequencePipe([
        FindTokensPipe("VERB/nsubj/*"),
        NamedEntityFilterPipe(),
        NamedEntityExtractorPipe()
    ]),
    FindTokensPipe("VERB"),
    AnyPipe([
        SequencePipe([
            FindTokensPipe("VBD/dobj/NNP"),
            AggregatePipe([
                NamedEntityFilterPipe("GPE"),
                NamedEntityFilterPipe("PERSON")
            ]),
            NamedEntityExtractorPipe()
        ]),
        SequencePipe([
            FindTokensPipe("VBD/**/*/pobj/NNP"),
            AggregatePipe([
                NamedEntityFilterPipe("LOC"),
                NamedEntityFilterPipe("PERSON")
            ]),
            NamedEntityExtractorPipe()
        ])
    ])
]

engine = PipelineEngine(pipes_structure, Context(doc), [0, 1, 2])
print(engine.process())

print("s")
