import tabula
import pdfplumber
import pandas as pd
import nltk
from nltk import word_tokenize
import re
from gensim.parsing.preprocessing import remove_stopwords
from service import Service
import docx
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch 
import pandas as pd
import numpy as np
import sent2vec
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from itertools import combinations
from operator import itemgetter
from tqdm import tqdm

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

pdf = pdfplumber.open(r"Service Catalogues\SQ15-000107_A (4).pdf")

def extract_all_tables(pdf):
    
    dictionary_of_tables = {} 
    for i in range(len(pdf.pages)):
        page = pdf.pages[i]
        for table in page.extract_tables():
            dictionary_of_tables['table'+str(i)] = pd.DataFrame(table[1:],columns=table[0],index=None)
    
    return dictionary_of_tables


dictionary_of_tables = extract_all_tables(pdf)


def filtering(string):
    
    string = string.replace('\n','')
    string = re.sub('[^a-zA-Z ]+', '', string)
    string = remove_stopwords(string) 
    #print(string)
    
    return string


def get_services(dictionary_of_tables):
    
    list_of_services = []
    for i in list(dictionary_of_tables.keys()):
        df = dictionary_of_tables[i]
        for index,row in df.iterrows():
            list_of_services.append(Service(str(row['Service']),filtering(str(row['Description']))))
    
    return list_of_services

list_of_services = get_services(dictionary_of_tables)


doc = docx.Document(r"Service Catalogues\Government-Goals.docx")
list_of_goals = [i.text for i in doc.paragraphs]

labels = ['Contradiction','Neutral','Entailment']
tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
#print(len(tokenizer))
model_rob = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")


def roberta_probability(sentence1,sentence2):
    
    #print(sentence1,sentence2)
    id1 = tokenizer.encode(sentence1,sentence2, padding = True, max_length=512, truncation='longest_first', return_tensors="pt")
    

    
    
    logits = model_rob.forward(id1, return_dict=True).logits

    p = torch.nn.functional.softmax(logits, dim=1)
    probs = p.tolist()[0]
    labels = ['Contradiction','Neutral','Entailment']
    
    result = {}
    for keys,values in zip(labels,probs):
        result[keys] = values
    
    result['Contradiction'] = result['Contradiction']*100
    result['Neutral'] = result['Neutral']*100
    result['Entailment']= result['Entailment']*100
    
    
    
    return result


module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" 
sbert_model = hub.load(module_url)
print ("module %s loaded" % module_url)


sentences = [i.description for i in list_of_services]

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

sentence_embeddings = sbert_model(sentences)
#len(sentence_embeddings)


def create_query_vecs(goals):
    
    query_vecs = []
    
    for i in goals:
        query_vecs.append(sbert_model([i])[0])
    
    return query_vecs

query_vecs = create_query_vecs(list_of_goals)


def create_similarity_dict(query_vecs,sentences):
    
    similarity_dict = {}
    index = 0
    for i in query_vecs:
        similarity_dict['Goal'+str(index)] = []
        index+=1
        for j in sentences:
            similarity_dict['Goal'+str(index-1)].append(cosine(i, sbert_model([j])[0]))

    return similarity_dict


similarity_dict = create_similarity_dict(query_vecs,sentences)

def get_topN_similarity(similarity_dict,N):
    top_similarity = {}
    for (key,value) in similarity_dict.items():
        servkey = (sorted(range(len(value)), key=lambda i:value[i])[:-N-1:-1])
        valsim = sorted(value)[:-N-1:-1]
        top_similarity[key] = [(i,j) for (i,j) in zip(servkey,valsim)]
        
    return top_similarity

goals_service_sim = get_topN_similarity(similarity_dict,15)


def postcondition_generator(similar_services,list_of_services,selection):
    
    combs = list(combinations(similar_services, selection))
    
    postconds = []
    
    for comb in combs:
        postcond = str()
        for index in comb:
            postcond += list_of_services[index[0]].description
            postcond += '. '
       
        postconds.append(postcond)
       
            
    
    return postconds,combs


def check_entailment(goal,list_of_postconds):
    
    
    
    result = roberta_probability(list_of_postconds[0],goal)
    
    pbar = tqdm(total = len(list_of_postconds),position=0, leave=True) 
    
    runs=1
    
    while (result['Entailment'] < 66 and runs <= len(list_of_postconds)):
        
        try:
            newresult = roberta_probability(list_of_postconds[runs],goal)
            
            if newresult['Entailment'] > result['Entailment']:
                result,runs = newresult,runs
            
            pbar.update(1)
            runs+=1
            
        except IndexError as error:
            print("Benchmark probability not reached")
            break
            
    pbar.close()
            
    return result,runs


def service_alignment(list_of_goals,goals_service_sim,list_of_services):
    
    alignment_dict = {}
    
    for (i,goal) in enumerate(list_of_goals):
        
        similar_services = goals_service_sim['Goal'+str(i)]
        #print(len(similar_services))
        
        print(f"\n\n-----Checking for the Goal {i}------\n\n")
        max_list = []
        
        for comb in range(1,len(similar_services)+1):
            
            print(f"\n\n--------Checking For the Combinations with length {comb}.--------\n\n")
            
            list_of_postconds,combs = postcondition_generator(similar_services,list_of_services,selection=comb)
            #print(len(list_of_postconds), comb)
            
            
            result,runs = check_entailment(goal,list_of_postconds)
            
            if (result['Entailment'] >= 66):
                
                alignment_dict['Goal'+str(i)] = (combs[runs-1],result['Entailment'])
                
                print(f"\n\nReached the benchmark probability at combination of {comb} and at the run {runs-1}\n\n")
                
                break
                
            else:
                
                max_list.append((combs[runs-1],result['Entailment']))
                
                alignment_dict['Goal'+str(i)] = max(max_list,key=itemgetter(1))
                
                continue
            
        
    
    return alignment_dict


alignment_dict = service_alignment(list_of_goals,goals_service_sim,list_of_services)

strategy_service_alignment = {}
for (key,value) in alignment_dict.items():
    
    goal = list_of_goals[int(key[-1])]
    services = [i[0] for i in value[0]]
    
    postconditions = ''
    
    for i in services:
        
        postconditions += list_of_services[i].description 
        postconditions += '. '
        
    probability = value[-1]
    
    aligned_services = [list_of_services[i].name for i in services]
    
    strategy_service_alignment[key] = {'Goal': goal,
                                       'Number of Services': len(aligned_services),
                                       'Aligned_services': aligned_services,
                                       'Postconditions' : postconditions,
                                       'Max_probability': probability}
    

alignment_df = pd.DataFrame(strategy_service_alignment)

alignment_df.to_csv(r'service_strategy_alignment.csv')


