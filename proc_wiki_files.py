import os
import json
import codecs
from collections import defaultdict
import unicodedata as uni

wiki_folder = 'wiki/wiki-pages-text'
dest_dir = "wiki/wiki-pages-proc"
files = os.listdir(wiki_folder)

index = []

import sqlite3
conn = sqlite3.connect('wiki/doc.db')
c = conn.cursor()
sql_create_projects_table = """ CREATE TABLE IF NOT EXISTS documents (
                                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                                        doc_id text,
                                        sen_id integer,
                                        text text
                                    ); """
c.execute(sql_create_projects_table)
indexing = """CREATE INDEX 'doc_id' ON documents('doc_id')"""
c.execute(indexing)

for file in files:
    print(file)
    with open(wiki_folder+"/"+file,'r') as data:
        for line in data:
            elem = line.strip().split(" ")
            try:
                doc = {}
                page_id = elem[0].replace("/","-SLH-")
                page_id = page_id.encode('utf8').decode('utf8')
                index.append(page_id)
                sen_id = int(elem[1])
                line = " ".join(elem[2:])
                line = line.replace('-LRB-','(')
                line = line.replace('-RRB-',')')
                line = line.replace('-LSB-','{')
                line = line.replace('-RSB-','}')
                line = line.replace('-COLON-',':')
                doc[sen_id] = line
            except ValueError:
                continue
            
            c.execute("INSERT INTO documents (doc_id,sen_id,text) VALUES (?,?,?)", (elem[0],sen_id,doc[sen_id]))
        
conn.commit()
conn.close()  
index = set(index)
with codecs.open('wiki/doc.txt', "w+","utf-8") as doc_file:
        for doc in index:
            doc_file.write(doc+'\n')
        
        doc_file.close()