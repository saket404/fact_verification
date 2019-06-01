"""
This file consists of the Document Retrieval and Sentence Retrieval code which writes the candidate sentences with its corresponding claim id to a json file for entail.py 
which performs the Textual Entailment. 
"""

import numpy as np
import time
from mediawiki import MediaWiki
import os
import json
import codecs
from collections import defaultdict
import stringdist
import nltk
import unicodedata
import wikipedia
import spacy
import neuralcoref
from util import check_parse, get_ner, doc_to_word, word_to_doc, get_NP
from allennlp.predictors.predictor import Predictor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import operator
import sqlite3
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
predictor = Predictor.from_path(
    "https://s3-us-west-2.amazonaws.com/allennlp/models/fine-grained-ner-model-elmo-2018.12.21.tar.gz")
predictor1 = Predictor.from_path(
    "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz")

# nlp = spacy.load('en')
# neuralcoref.add_to_pipe(nlp, greedyness=0.5)

# ------------ Get names of all documents -------------
index = []
with codecs.open('doc.txt', "r+", "utf-8") as doc_file:
    content = doc_file.readlines()
    for doc in content:
        doc = doc.strip()
        index.append(doc)

doc_ent = [doc_to_word(word) for word in index]
index = []
# ----------- Document Retrieval component ---------------


def wiki(word):
    res = wikipedia.search(word, results=1)
    if res:
        return res[0]
    else:
        return None


def doc_ret(sen):
    org = sen
    claim = get_ner(sen, predictor)
    # print(f'Claim NER: {claim}')
    c_doc = []
    # TO handle entity ambiguation
    get_noun = get_NP(org, predictor1)
    if get_noun:
        if get_noun in doc_ent:
            c_doc.append(get_noun)
        else:
            get_noun = wiki(get_noun)
            if get_noun in doc_ent:
                c_doc.append(get_noun)

    # For each entity find the document.
    for c_ent in claim:
        result = ""
        c_ent = unicodedata.normalize('NFC', c_ent)
        if c_ent in doc_ent and c_ent not in c_doc:
            result = c_ent
            c_doc.append(result)

        result = wiki(c_ent)
        if result in doc_ent and result not in c_doc:
            c_doc.append(result)

    # TO handle entity ambiguation
    result = check_parse(org, predictor1)
    for w in result:
        parse = wiki(w)
        if parse and parse in doc_ent and parse not in c_doc:
            c_doc.append(parse)

    # If NER cant extract entities and NP not extractable
    if not result and not claim and not get_noun:
        result = wiki(org)
        if result and result in doc_ent:
            c_doc.append(result)

    final = [word_to_doc(i) for i in list(set(c_doc))]
    return final

# ----------- Sentence Retrieval component ---------------


def coref(sentences):
    count = 0
    first = ""
    check_doc = ""
    output = {}
    sent = []
    for sen in sentences:
        sent.append((sen[0], sen[1], sen[2], sen[3]))
        if check_doc != sen[1]:
            count = 0

        if count == 0:
            check_doc = sen[1]
            first = doc_to_word(sen[1]) + ' .'
            output[sen[1]+" "+str(sen[2])] = sen[3]
            count += 1
        else:
            text = sen[3].replace('.', '')
            concat = first + " " + text + "."
            cor = nlp(concat)
            res = cor._.coref_resolved
            update = res.split('.')[1] + "."
            output[sen[1] + " " + str(sen[2])] = update

    return sent, output


def lemmatize(word):
    lemma = lemmatizer.lemmatize(word, 'v')
    if lemma == word:
        lemma = lemmatizer.lemmatize(word, 'n')
    return lemma


def sen_retrieval(docs, claim):
    if not docs:
        return None
    claim = claim.replace('.', '')
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    conn = sqlite3.connect('doc.db')
    c = conn.cursor()

    texts = []
    index = []
    doc_sen = {}

    placeholder = '?'
    placeholders = ', '.join(placeholder for unused in docs)
    query = 'SELECT * FROM documents WHERE doc_id IN (%s)' % placeholders

    sentences = c.execute(query, docs)
    sentences, coref_sentences = coref(sentences)
    for i in sentences:
        index.append(str(i[1]) + ' ' + str(i[2]))
        sen = [lemmatize(i).lower() for i in i[3].split()]
        texts.append(" ".join(sen))
        doc_sen[str(i[1]) + ' ' + str(i[2])] = i[3]

    c.close()

    matrix = vectorizer.fit_transform(texts)
    claim = [lemmatize(i).lower() for i in claim.split()]
    query = vectorizer.transform([" ".join(claim)])[0]
    cosineSimilarities = cosine_similarity(query, matrix).flatten()
    idx = sorted(range(len(cosineSimilarities)),
                 key=lambda i: cosineSimilarities[i], reverse=True)[:3]
    rel = [index[i] for i in idx]
    relevant_sen = [[i.split(' ')[0], int(i.split(' ')[1]), doc_sen[i]]
                    for i in rel]

    return(relevant_sen)


# ----------- Main component ---------------
corpus_folder = 'wiki/wiki-pages-text'
db_path = 'wiki/doc.db'
claim_file = "devset.json"

with codecs.open(claim_file, 'r+', 'utf-8') as test_file:
    data = json.load(test_file)

start = time. time()
to_entail = {}
count = 1
with codecs.open('temp.json', 'w+', 'utf-8') as temp_file:
    for i in data.keys():
        if count in [500, 1000, 5000, 8000, 10000]:
            print(count)
        # print('==================')
        try:
            rel_docs = doc_ret(data[i]['claim'])
        except Exception:
            print(f"Error at {i}")
            rel_docs = []
        dummy = {}
        dummy['claim'] = data[i]['claim']
        dummy['candidate'] = sen_retrieval(rel_docs, data[i]['claim'])
        to_entail[i] = dummy
        count += 1
        # print('=====================')

    json.dump(to_entail, temp_file, indent=2)


print('############# Stats ###########')
end = time.time()
print(end - start)
