"""
This file consists of Textual Entailment component where it takes the json file output from main.py (candidate sentences) and 
performs the textual entailment and writes it to a json file for evaluation.
"""
import numpy as np
import time
import os
import json
import codecs
import unicodedata
from util import check_parse, get_ner, doc_to_word, word_to_doc, get_NP
from allennlp.predictors.predictor import Predictor


predictor2 = Predictor.from_path(
    "https://s3-us-west-2.amazonaws.com/allennlp/models/decomposable-attention-elmo-2018.02.19.tar.gz")
# ----------- RTE component ---------------


def entailment(rel, claim):
    ret = {0: 'SUPPORTS', 1: 'REFUTES', 2: 'NOT ENOUGH INFO'}
    ret_sen = rel
    if rel:
        concat = [sen[2] for sen in rel]
        concat = " ".join(concat)
        out2 = predictor2.predict(hypothesis=claim, premise=concat)
        a = np.array(out2['label_probs'])

        label = ret[np.argmax(a)]
        if label == ret[2]:
            evidence = []
        else:
            evidence = [[sen[0], sen[1]] for sen in ret_sen]

    else:
        label = ret[2]
        evidence = []
    return label, evidence


start = time. time()
with codecs.open('temp.json', 'r+', 'utf-8') as test_file:
    data2 = json.load(test_file)
count = 0
write_file = codecs.open("testoutput.json", "w+", 'utf-8')
wr = {}
for i in data2.keys():
    if count in [10, 100, 500, 1000, 2000, 3000, 5000, 8000, 10000, 12000]:
        print(count)
        end = time.time()
        print(end - start)

    label, predicted = entailment(data2[i]['candidate'], data2[i]['claim'])
    if predicted:
        for j in range(len(predicted)-1):
            predicted[j][0] = unicodedata.normalize('NFD', predicted[j][0])
    out = {}
    out['claim'] = data2[i]['claim']
    out['label'] = label
    out['evidence'] = predicted
    wr[i] = out
    count += 1


json.dump(wr, write_file, indent=2)
print('############# Stats ###########')
end = time.time()
print(end - start)
