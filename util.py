"""
This file consists 5 functions and are utilities used by the main components:

get_NP: which gets the Noun phrase before the first verb phrase.
check_parse: is a function which appends WORK_OF_ART tokens to entities for wikipedia search.
get_ner: is the function which implements the extraction of NER using AllenNLP and combines multi-word entities as one
doc_to_word: is converting document names to normal wiki pages format.
word_to_doc: is the vice versa of doc_to_word. 
"""

WORK_OF_ART = {'film': ' film', 'movie': ' film', 'series': ' TV',
               'show': ' TV', 'song': ' song', 'album': ' album',
               'season': ' TV'}
phrase = ['directed by', 'based on']


def get_NP(claim, predictor, check=''):
    result = predictor.predict(claim)
    entity = ''
    tree = result['hierplane_tree']['root']
    if 'children' in tree.keys():
        children = tree['children']
    if children[0]['nodeType'] == 'NP' and children[1]['nodeType'] == 'VP':
        if check != '' and check in children[1]['word']:
            entity = children[0]['word'] + WORK_OF_ART['film']
        else:
            entity = children[0]['word']

    return entity


def check_parse(claim, predictor):

    claim = claim.replace(".", "")
    ent = []
    if any(i in claim for i in WORK_OF_ART.keys()):
        param = [WORK_OF_ART[i] for i in WORK_OF_ART.keys() if i in claim]
        result = get_NP(claim, predictor)
        if result:
            ent.append(result+param[0])
    else:
        param = [i for i in phrase if i in claim]
        if param:
            result = get_NP(claim, predictor, check=param[0])

    return list(set(ent))


def get_ner(sentence, predictor):
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
                    add = [WORK_OF_ART[i]
                           for i in WORK_OF_ART.keys() if i in org]
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
                    add = [WORK_OF_ART[i]
                           for i in WORK_OF_ART.keys() if i in org]
                    word = words[i] + add[0]
                else:
                    word = words[i]
            else:
                word = words[i]
            docs.append(word)

    return docs


def doc_to_word(word):
    word = word.replace("-SLH-", "/")
    word = word.replace("_", " ")
    word = word.replace("-LRB-", "(")
    word = word.replace("-RRB-", ")")
    word = word.replace("-COLON-", ":")
    return(word)


def word_to_doc(word):
    word = word.replace("/", "-SLH-")
    word = word.replace(" ", "_")
    word = word.replace("(", "-LRB-")
    word = word.replace(")", "-RRB-")
    word = word.replace(":", "-COLON-")
    return(word)
