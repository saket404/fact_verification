WORK_OF_ART = {'film':' film','movie':' film','serie':' TV','show':' TV'}
tense = ['is','was','were','are','had','has']

def check_parse(claim):
    
    claim = claim.replace(".","")
    ent = []
    if any(i in claim for i in WORK_OF_ART.keys()):
        param = [WORK_OF_ART[i] for i in WORK_OF_ART.keys() if i in claim]
        tense_s = [i for i in tense if i in claim.split()]
        if tense_s:
            ent.append(claim.split(tense_s[0])[0]+param[0])
    if 'directed by' in claim:
        tense_s = [i for i in tense if i in claim.split()]
        if tense_s:
            ent.append(claim.split(tense_s[0])[0]+WORK_OF_ART['film'])
    if 'premiered' in claim:
        ent.append(claim.split('premiered')[0]+WORK_OF_ART['film'])
    if 'based on' in claim:
        tense_s = [i for i in tense if i in claim.split()]
        if tense_s:
            ent.append(claim.split(tense_s[0])[0]+WORK_OF_ART['film'])
    
    return list(set(ent))

def get_ner(sentence,predictor):
    org = sentence
    result = predictor.predict(sentence)
    words = []
    tags = []
    for word, tag in zip(result["words"], result["tags"]):
        if tag != "O":
            words.append(word)
            tags.append(tag)

    docs = []
    prev = []
    for i in range(len(words)):
        if "DATE" in tags[i] or "CARDINAL" in tags[i] or "ORDINAL" in tags[i]:
            continue
        elif 'B-' in tags[i]:
            prev.append(words[i])
        elif 'I-' in tags[i]:
            prev.append(words[i])
        elif 'L-' in tags[i]:
            prev.append(words[i])
            if 'WORK_OF_ART' in tags[i]:
                if any(i in org for i in WORK_OF_ART.keys()):
                    add = [WORK_OF_ART[i] for i in WORK_OF_ART.keys() if i in org]
                    word = " ".join(prev) + add[0]
                else:
                    word = " ".join(prev)
            else:
                word = " ".join(prev)
            docs.append(word)
            prev = []
        else:
            if 'WORK_OF_ART' in tags[i]:
                if any(i in org for i in WORK_OF_ART.keys()):
                    add = [WORK_OF_ART[i] for i in WORK_OF_ART.keys() if i in org]
                    word = words[i] + add[0]
                else:
                    word = words[i]
            else:
                word  = words[i]
            docs.append(word)

    return docs

def doc_to_word(word):  
    word = word.replace("-SLH-","/")
    word = word.replace("_"," ")
    word = word.replace("-LRB-","(")
    word = word.replace("-RRB-",")")
    return(word)

def word_to_doc(word):  
    word = word.replace("/","-SLH-")
    word = word.replace(" ","_")
    word = word.replace("(","-LRB-")
    word = word.replace(")","-RRB-")
    return(word)