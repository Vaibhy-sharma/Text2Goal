import tabula
import pdfplumber
import pandas as pd
import nltk
from nltk.corpus import reuters
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import re

from service import Service

pdf = pdfplumber.open(r"Service Catalogues\SQ15-000107_A (4).pdf")

def filtering(string):
    
    string = string.replace('\n','')
    string = re.sub('[^a-zA-Z ]+', '', string)
    
    return string

def extract_all_tables(pdf):
    
    dictionary_of_tables = {} 
    for i in range(len(pdf.pages)):
        page = pdf.pages[i]
        for table in page.extract_tables():
            dictionary_of_tables['table'+str(i)] = pd.DataFrame(table[1:],columns=table[0],index=None)
    
    return dictionary_of_tables


def get_list_of_services(dictionary_of_tables):

	list_of_services = []
	for i in list(dictionary_of_tables.keys()):
		df = dictionary_of_tables[i]
		for index,row in df.iterrows():
			list_of_services.append(Service(filtering(str(row['Service'])),filtering(str(row['Description']))))


	return list_of_services
        
                


