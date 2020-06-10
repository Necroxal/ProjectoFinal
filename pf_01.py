# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 13:03:17 2020

@author: luisr
"""
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import re
from math import log

main_dir ='C:/Users/luisr/Desktop/Archivos/DICIS/Materias Actuales/MD/Face/'
D=main_dir+'archivo.txt'
d=main_dir+'prueba.txt'
C=('H','M')

def voc(D):
    stop_words = stopwords.words('spanish')
    file=D
    corpus = []
    with open(file, 'r', encoding='utf-8') as content_file:
        for line in content_file:
            line = line.rstrip().lower()
            words = re.findall('[a-záéíóúñ]+', line)
            text = ' '.join(word for word in words if word not in stop_words and len(word)>3)
            corpus.append(text)      
        vec = CountVectorizer()
        tdm = vec.fit_transform(corpus)
        vocabulary = vec.vocabulary_ 
        lista= list(vocabulary.keys())
    return(lista)

def count(c,D):
    file =D
    cont=0
    with open(file, 'r', encoding='utf-8') as content_file:
        for line in content_file:
            if(line[0]==c):
                cont+=1 #cuenta cuantas veces aparece una palabra en una lista y las agrega a otra lsita
    return (cont)
        
def TrainB(C,D):
    file=D
    V=voc(D)
    prior=[]
    N=sum(1 for line in open (file, encoding='utf-8'))
    Nct=0
    contprob=[]
    for v in range(len(V)):
        contprob.append([])
    for c in range(len(C)):
       Nc=count(C[c],file)
       prior.append(Nc/N)
       for t in range(len(V)):
           with open(file, 'r', encoding='utf-8') as content_file:
               for line in content_file:
                   if(line[0]==C[c]):
                       line=line.lower()
                       cuenta = line.count(V[t])
                       if(cuenta>0):
                           Nct+=1              
           contprob[t].append((Nct+1)/(Nc+2))
           Nct=0
    return(V,prior,contprob)
           
def applyB(C,V,prior,contprob,d):
    Vd=[]
    score=[]
    with open(d,'r',encoding='utf-8') as content_file:
        text=content_file.read()
    text=text.lower()
    for p in V:
        cont=text.count(p)
        if cont>0:
            Vd.append(p)
    for c in range(len(C)):
        if prior[c]==0:
            score.append(0)
        else:
            score.append(log(prior[c]))
        for t in V:
            if t in Vd:
                score[c]+=log(contprob[V.index(t)][c])  
            else:
                score[c]+=log(1-contprob[V.index(t)][c])
    print(score)
    return(C[score.index(max(score))])
                
                
val=TrainB(C,D)
#print(val[2])
result=applyB(C,val[0],val[1],val[2],d)
print(result)