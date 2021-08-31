import sys
import os
import xml.etree.cElementTree as ET
import math
import json
import string
import nltk
#from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

#------------------------------create_index------------------------------

def get_tf(word, doc, d):
    return d[word][doc]

def get_idf(word, d):
    return d[word]["idf"]

def computing_vector_lengths(d):  #for all documents
    vectors_lengths=dict()
    for word in d:
        idf=get_idf(word, d)
        for doc in d[word]:
            if doc!="idf":
                tf=get_tf(word, doc, d)
                w=idf*tf
                if doc not in vectors_lengths:
                    vectors_lengths[doc]=(w)**2
                else:
                    vectors_lengths[doc]=vectors_lengths[doc]+(w)**2
    for x in vectors_lengths:
        vectors_lengths[x]=(vectors_lengths[x])**(0.5)
    d["###vectors_lengths###"]=vectors_lengths

def computing_idf_and_divide_tf_in_max_value(d, number_of_docs, max_value):
    for word in d:
        df=len(d[word])
        idf=math.log(number_of_docs/df)
        for doc in d[word]:
            d[word][doc]=d[word][doc]/(max_value[doc])
        d[word]["idf"]=idf
        

def inverted_index(record_num, text, d, max_value):
    for word in text:
        if word not in d:
            d[word]=dict()
        if record_num not in d[word]:
            d[word][record_num]=1
        else:
            d[word][record_num]=d[word][record_num]+1
        if record_num not in max_value:
            max_value[record_num]=d[word][record_num]
        else:
            if max_value[record_num]<d[word][record_num]:
                max_value[record_num]=d[word][record_num]
                
def tokenization_and_removing_stopwords(record_num, text):
    #stop_words = set(stopwords.words('english'))
    stop_words={'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be','some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than'} 
    punct = string.punctuation
    punct.replace("-","")
    punct_arr=[0]*32
    for i in range(0,32):
        punct_arr[i]=punct[i]
    lower_case_text=text.lower()
    word_tokens=word_tokenize(lower_case_text)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    filtered_sentence = [w for w in filtered_sentence if not w in punct_arr]
    filtered_sentence = [w for w in filtered_sentence if not w.isdigit()]
    porter_stemmer= PorterStemmer()
    filtered_sentence= [porter_stemmer.stem(x) for x in filtered_sentence]
    return (record_num, filtered_sentence)

def create_index(corpus_directory, d):
    entries = os.listdir(corpus_directory)
    number_of_docs=0
    max_value=dict()
    for entry in entries:
        if entry[-3:]=="xml" and entry[-5]!="y":
            tree = ET.parse(corpus_directory+"/"+entry)
            root=tree.getroot()
            for record in root.findall("RECORD"):
                number_of_docs=number_of_docs+1
                title = record.find('TITLE').text
                extract= record.find('EXTRACT')
                if extract!=None:
                    extract=extract.text
                abstract = record.find('ABSTRACT')
                if abstract!=None:
                   abstract=abstract.text
                if extract!=None and abstract!=None:
                   a=tokenization_and_removing_stopwords(number_of_docs, title+" "+extract+" "+abstract)
                elif extract==None and abstract!=None:
                    a=tokenization_and_removing_stopwords(number_of_docs, title+" "+abstract)
                elif extract!=None and abstract==None:
                    a=tokenization_and_removing_stopwords(number_of_docs, title+" "+extract)
                else:
                    a=tokenization_and_removing_stopwords(number_of_docs, title)
                inverted_index(a[0], a[1], d, max_value)
    computing_idf_and_divide_tf_in_max_value(d, number_of_docs, max_value)
    computing_vector_lengths(d)
    with open('vsm_inverted_index.json', 'w') as fp:
        json.dump(d, fp)
        
#------------------------------query------------------------------

def creating_query_dict(question):
    max_value=0
    query_dict=dict()
    for word in question:
        if word not in query_dict:
            query_dict[word]=1
        else:
            query_dict[word]=query_dict[word]+1
        if max_value<query_dict[word]:
            max_value=query_dict[word]
    for word in query_dict:
        query_dict[word]=query_dict[word]/max_value
    return query_dict

def get_W_query(word, d, query_dict):
    I=get_idf(word, d)
    K=query_dict[word] 
    W_query=I*K # weight of token word in the query
    return W_query

def computing_query_length(question, query_dict, d):
    query_length=0
    for word in question:
        if word in d:
            W_query=get_W_query(word, d, query_dict)
            query_length=query_length+(W_query)**2
    query_length=query_length**(0.5)
    return query_length

def query(question, d, i):
    documents_scores=dict()
    question=tokenization_and_removing_stopwords("query", question)[1]
    query_dict=creating_query_dict(question)
    query_length=computing_query_length(question, query_dict, d)
    for word in question:
        if word in d:
            I=get_idf(word, d)
            K=query_dict[word] 
            W_query=I*K # weight of token word in the query
            list_of_docs=d[word]
            for doc in list_of_docs:
                if doc!="idf":
                    tf=get_tf(word, doc, d)
                    W_doc=I*tf
                    doc_length=d["###vectors_lengths###"][doc]
                    w=(W_query*W_doc)/(query_length*doc_length)
                    if doc not in documents_scores:
                        documents_scores[doc]=w
                    else:
                        documents_scores[doc]=documents_scores[doc]+w
    documents_scores_sorted=[k for k in sorted(documents_scores.items(), key=lambda item: item[1], reverse=True)]
    documents_scores_sorted=[k[0] for k in documents_scores_sorted if k[1]>=0.075]
    if (i==0):
        f=open("ranked_query_docs.txt", 'w', encoding="utf-8")
        for t in documents_scores_sorted:
            f.write(t+'\n')
    else:
        f=open("ranked_query_docs"+str(i)+".txt", 'w', encoding="utf-8")
        for t in documents_scores_sorted:
            f.write(t+'\n')

#------------------------------evaluation------------------------------
   
def result(corpus_directory, num_query):    
    entries = os.listdir(corpus_directory)
    for entry in entries:
        if entry=="cfquery.xml":
            tree = ET.parse(corpus_directory+"/"+entry)
            root=tree.getroot()
            for query in root.findall("QUERY"):
                query_number= int(query.find("QueryNumber").text) 
                query_text = query.find('QueryText').text
                results= int(query.find('Results').text)
                records= query.find('Records')
                s=set()
                for item in records.findall('Item'):
                    s.add(int(item.text))
                if num_query==query_number:
                    return((query_number, query_text, results, s))

def counter(retrieved_documents, relevant_documents):
    counter=0
    for doc in retrieved_documents:
        if doc in relevant_documents:
            counter=counter+1
    return counter
    
def evaluation(corpus_directory, text_file, num_query):
    a=result(corpus_directory, num_query)
    relevant_documents=a[3]
    f = open(text_file, "r") 
    retrieved_documents=f.readlines()
    retrieved_documents=[int(x[:-1]) for x in retrieved_documents]
    Counter=counter(retrieved_documents, relevant_documents)
    Precision=(Counter)/(len(retrieved_documents))
    Recall=(Counter)/(len(relevant_documents))
    F=(2*Precision*Recall)/(Precision+Recall)
    print(num_query, Precision, Recall, F)
    return F
    
def queries(corpus_directory, d):
    for i in range(1, 101):
        if (i!=93):
            question=result(corpus_directory, i)[1]
            query(question, d, i)
            
def results(corpus_directory):
    everage_F=0
    for i in range(1, 101):
        if (i!=93): 
            everage_F=everage_F+evaluation(corpus_directory, "ranked_query_docs"+str(i)+".txt", i)
    print("everage_F", everage_F/99)
       
if __name__ == "__main__":
    d={}
    if sys.argv[1]=="create_index": #create_index C:\Users\Tali\Desktop\wdm\ex3\cfc-xml_corrected
        create_index(sys.argv[2], d)
    else:
        if sys.argv[1]=="query":   #query C:\Users\Tali\Desktop\wdm\ex3\vsm_inverted_index.json "What are the effects of calcium on the physical properties of mucus from CF patients?"
            with open(sys.argv[2]) as json_file:
                d = json.load(json_file)
            query(sys.argv[3], d, 0)
        #delete
        else:
            if sys.argv[1]=="queries":  #queries C:\Users\Tali\Desktop\wdm\ex3\vsm_inverted_index.json C:\Users\Tali\Desktop\wdm\ex3\cfc-xml_corrected
                with open(sys.argv[2]) as json_file:
                    d = json.load(json_file)
                queries(sys.argv[3], d)
            else:
                if sys.argv[1]=="results":  #results C:\Users\Tali\Desktop\wdm\ex3\cfc-xml_corrected
                    results(sys.argv[2])
                else:
                    evaluation(sys.argv[1], sys.argv[2], int(sys.argv[3])) #C:\Users\Tali\Desktop\wdm\ex3\cfc-xml_corrected C:\Users\Tali\Desktop\wdm\ex3\ranked_query_docs.txt 1
    
