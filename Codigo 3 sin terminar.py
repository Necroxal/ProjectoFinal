#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 15:09:15 2020

@author: jonathanestrella
"""
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import re

main_dir = 'C:/Users/luisr/Desktop/Archivos/DICIS/Materias Actuales/MD/Face/'
file = main_dir+'archivo2.txt'

def norm(file):
    stop_words = stopwords.words('spanish')
    corpus = []
    with open(file, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.rstrip().lower()
            words = re.findall('[a-záéíóúñ]+', line)
            text = ' '.join(word for word in words if word not in stop_words and len(word) > 3)
            corpus.append(text)
            
        vec = TfidfVectorizer(norm = 'l2')
        tdm= vec.fit_transform(corpus)
    return tdm

def valm(i):
    i.sort()
    vm=i[-2]
    m=0
    m=i.index(vm)
    return m


C=[]
I=[]
N=sum(1 for line in open (file, encoding='utf-8'))
tdm1=norm(file)
for n in range(N):
    C.append([])
    I.append(1)
    for i in range(N):
        x=(cosine_similarity(tdm1[n],tdm1[i]))
        C[n].append(x[0][0])
A=[]
argM={}
for k in range(N-1):
    i=k
    m=valm(C[i])
    argM[(i,m)]=C[i][m]
    A.append(argM[(i,m)])
    for j in range(N):
        C[i][j]=(cosine_similarity(,tdm1[i]))