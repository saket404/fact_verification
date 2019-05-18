import os
import json
import codecs
from collections import defaultdict
import nltk
import unicodedata 
import wikipedia
from util import check_parse, get_ner, doc_to_word, word_to_doc
from allennlp.predictors.predictor import Predictor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import operator
import sqlite3
predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/fine-grained-ner-model-elmo-2018.12.21.tar.gz")


#------------ Get names of all documents -------------
index = []
with codecs.open('wiki/doc.txt', "r+","utf-8") as doc_file:
    content = doc_file.readlines()
    for doc in content:
        doc = doc.strip()
        index.append(doc)

doc_ent = []
for word in index:
    word = doc_to_word(word)
    doc_ent.append(word)
    
#----------- Document Retrieval component ---------------
def wiki(word):
  res = wikipedia.search(word,1)
  if res:
    return res[0]
  else:
    return None

def doc_ret(sen):
    org = sen
    claim = get_ner(sen,predictor)
    print(f'Claim NER: {claim}')
    c_doc = []
    for c_ent in claim:
        result = ""
        c_ent = unicodedata.normalize('NFC',c_ent)
        if c_ent in doc_ent:
            result = c_ent
            result = word_to_doc(result)
            c_doc.append(result) 

        result = wiki(c_ent)
        if result in doc_ent:
            result = word_to_doc(result)
            c_doc.append(result)

        if not result:
            low = 1
            low_d = ''
            for en in doc_ent:
                d = stringdist.levenshtein_norm(en,c_ent)
                if  d <= low:
                    low = d
                    low_d = en 
            low_d = word_to_doc(low_d)
            c_doc.append(low_d)
        
    result = check_parse(org)
    for w in result:
      parse = wiki(w)
      if parse and parse in doc_ent:
        parse = word_to_doc(parse)
        c_doc.append(parse)
    
      
    if not result and not claim:
      tense = ['is','was','were','are','had','has']
      tense_s = [i for i in tense if i in org.split()]
      if tense_s:
        result = wiki(org.split(tense_s[0])[0])
      else:
        result = wiki(org)
      
        if result and result in doc_ent:
              result = word_to_doc(result)
              c_doc.append(result)
  

    return list(set(c_doc))

#----------- Sentence Retrieval component ---------------
def sen_retrieval(docs,claim):
    claim = claim.replace('.','')
    vectorizer = TfidfVectorizer(stop_words = 'english',ngram_range = (2,3))
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
    #relevant_sen = [(i.split(' ')[0],int(i.split(' ')[1]),doc_sen[i]) for i in rel]
    
    return(rel)
    

corpus_folder = 'wiki/wiki-pages-text'
db_path = 'wiki/doc.db'
claim_file = "devset.json"

with codecs.open(claim_file,'r+','utf-8') as test_file:
    data = json.load(test_file)


count = 10
for i in data.keys():
        if count != 0:
            print('==================')
            print(f'Claim: {data[i]["claim"]}')
            rel_docs = doc_ret(data[i]['claim'])
            print(f'Relevant Docs: {rel_docs}')
            print(sen_retrieval(rel_docs,data[i]['claim']))
            print('----------')
            print(data[i]['evidence'])
            print('==================')
            count = count -1
        else:
            break
    