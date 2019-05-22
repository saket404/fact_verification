import os
import json
import codecs
from collections import defaultdict
import stringdist
import nltk
import unicodedata 
import wikipedia
from util import check_parse, get_ner, doc_to_word, word_to_doc, get_NP
from allennlp.predictors.predictor import Predictor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import operator
import sqlite3
predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/fine-grained-ner-model-elmo-2018.12.21.tar.gz")
predictor1 = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz")
from mediawiki import MediaWiki
wikipedia = MediaWiki()

#------------ Get names of all documents -------------
index = []
with codecs.open('wiki/doc.txt', "r+","utf-8") as doc_file:
    content = doc_file.readlines()
    for doc in content:
        doc = doc.strip()
        index.append(doc)

doc_ent = [doc_to_word(word) for word in index]
#----------- Document Retrieval component ---------------
def wiki(word):
  res = wikipedia.search(word,results = 1)
  if res:
    return res[0]
  else:
    return None

def doc_ret(sen):
    org = sen
    claim = get_ner(sen,predictor)
    print(f'Claim NER: {claim}')
    c_doc = []
    get_noun = get_NP(org,predictor1)
    if get_noun:
        if get_noun in doc_ent:
            c_doc.append(get_noun)
        else:
            get_noun = wiki(get_noun)
            if get_noun in doc_ent:
                c_doc.append(get_noun)

    for c_ent in claim:
        result = ""
        c_ent = unicodedata.normalize('NFC',c_ent)
        if c_ent in doc_ent and c_ent not in c_doc:
            result = c_ent
            c_doc.append(result) 

        result = wiki(c_ent)
        if result in doc_ent and result not in c_doc:
            c_doc.append(result)
    
    # TO handle entity ambiguation
    result = check_parse(org,predictor1)
    for w in result:
      parse = wiki(w)
      if parse and parse in doc_ent and parse not in c_doc:
        c_doc.append(parse)
    
    
    # If NER cant extract entities
    if not result and not claim and not get_noun:
        result = wiki(org)
        if result and result in doc_ent:
            c_doc.append(result)
  
    final = [word_to_doc(i) for i in list(set(c_doc))]
    return final

#----------- Sentence Retrieval component ---------------
def sen_retrieval(docs,claim):
    if not docs:
        return None
    claim = claim.replace('.','')
    vectorizer = TfidfVectorizer(stop_words = 'english',ngram_range = (1,3))
    conn = sqlite3.connect('wiki/doc.db')
    c = conn.cursor()
    
    texts = []
    index = []
    doc_sen = {} 

    placeholder= '?'
    placeholders= ', '.join(placeholder for unused in docs)
    query= 'SELECT * FROM documents WHERE doc_id IN (%s)' % placeholders

    sentences = c.execute(query,docs)
    for i in sentences:
        index.append(str(i[1]) + ' ' + str(i[2]))
        texts.append(i[3])
        doc_sen[str(i[1]) + ' ' + str(i[2])] = i[3]

    c.close()
    
    matrix = vectorizer.fit_transform(texts)
    query = vectorizer.transform([claim])[0]
    cosineSimilarities = cosine_similarity(query, matrix).flatten()
    idx = sorted(range(len(cosineSimilarities)), key=lambda i: cosineSimilarities[i], reverse=True)[:5]
    rel = [index[i] for i in idx]
    relevant_sen = [(i.split(' ')[0],int(i.split(' ')[1]),doc_sen[i]) for i in rel]
    
    return(rel)
    

#----------- Main component ---------------
corpus_folder = 'wiki/wiki-pages-text'
db_path = 'wiki/doc.db'
claim_file = "devset.json"

with codecs.open(claim_file,'r+','utf-8') as test_file:
    data = json.load(test_file)

import time
start = time. time()

count = 10
t_h = 0
t_m = 0
for i in data.keys():
        if count != 0:
            print('==================')
            print(f'Claim: {data[i]["claim"]}')
            rel_docs = doc_ret(data[i]['claim'])
            print(f'Relevant Docs: {rel_docs}')
            candidate = sen_retrieval(rel_docs,data[i]['claim'])
            print(candidate)
            
            candidate = [unicodedata.normalize('NFD',i) for i in candidate]
            evidence = data[i]['evidence']

            hit = 0
            miss = 0
            for i in evidence:
                actual = ' '.join(str(ii) for ii in i)
                if candidate:
                    if actual in candidate:
                        hit += 1 
                    else:
                        miss += 1
                        
            if miss == 0 and hit != 0:
                t_h += 1
            elif miss != 0:
                t_m += 1
            else:
                pass
            # print(f'Candidate sen: {candidate}')
            print(f'Atual sen: {evidence}')
            print(f'Hit: {hit}  Miss: {miss}')
            print('==================')
            count = count -1
        else:
            break
    
print('############# Stats ###########')
print(f'Total count = {t_h + t_m}')
print(f'Total Hit = {t_h}')
print(f'Total Miss = {t_m}')
print(f'Accuracy = {t_h/(t_h + t_m)}')
end = time.time()
print(end - start)
    
