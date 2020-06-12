# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 13:03:17 2020

@author: luisr
"""

#importar librerias
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import re
from math import log

#abrimos archivo
main_dir ='C:/Users/luisr/Desktop/Archivos/DICIS/Materias Actuales/MD/Face/'
D=main_dir+'archivo.txt'
d=main_dir+'prueba.txt'
C=('H','M')

#funcion para obtener vocabulario
def voc(file):
    stop_words = stopwords.words('spanish')
    corpus = []
    #recorremos el documento por linea y filtramos las palabras
    with open(file, 'r', encoding='utf-8') as content_file:
        for line in content_file:
            line = line.rstrip().lower()#pasar a minusculas
            words = re.findall('[a-záéíóúñ]+', line)#filtro de palabras 
            text = ' '.join(word for word in words if word not in stop_words and len(word)>3)#palabras mayores a 3
            corpus.append(text)#juntamos cada linea
        vec = CountVectorizer()#Convierta una colección de documentos de texto en una matriz de recuentos de tokens
        tdm = vec.fit_transform(corpus)
        vocabulary = vec.vocabulary_ #crea diccionario contando la aparicion de palabras
        lista= list(vocabulary.keys()) #obtenemos las llaves del diccionario
    return(lista) #regresa el vocabulario

#funcion que cuenta cuantos archivos son de hombre y cuantos de mujer
def count(c,file):
    cont=0#contador de archivos
    with open(file, 'r', encoding='utf-8') as content_file:#abrimos archivo por lineas
        for line in content_file:
            if(line[0]==c):#si comienza por la letra igual al sexo, se aplica
                cont+=1 #suma 1
    return (cont)

#entrenador de programa
def TrainB(C,file):
    V=voc(D)#obtenemos el vocabulario de nuestro archivo con oublicaciones
    prior=[]
    N=sum(1 for line in open (file, encoding='utf-8'))#Cuenta numero de lineas
    Nct=0
    contprob=[]
    for v in range(len(V)):#creamos espacios para asignar valores de hombre y mujer
        contprob.append([])#creamos una lista dentro de otra para agregar valores de probabilidad de hombre y mujer
    for c in range(len(C)):#recorremos primero para hombre y despues mujer
       Nc=count(C[c],file)#cuenta cuantos archivos del genero en cuestion existen en nuestro documento
       prior.append(Nc/N)#numero de documentos de un genero sobre el numero total de documentos
       for t in range(len(V)):#recorremos el vocabulario
           with open(file, 'r', encoding='utf-8') as content_file:#abrimos el archivo y recorremos por lineas
               for line in content_file:
                   if(line[0]==C[c]):#nos aseguramos de que exista al menos 1 documento para cada genero
                       line=line.lower()#pasamos la linea en minusculas
                       cuenta = line.count(V[t])#contamos si el archivo contiene la palabra del vocabulario
                       if(cuenta>0):#si lo contiene se suma 1 a Nct
                           Nct+=1              
           contprob[t].append((Nct+1)/(Nc+2))#calculamos la probabilidad de aparicion para cada genero y lo agregamos a nuestra lista de listas
           Nct=0#reinciamos Nct
    return(V,prior,contprob)

#funcion de aplicacion a documento
def applyB(C,V,prior,contprob,d):
    Vd=[]#creamos diccionario del nuevo documento en base al anterior
    score=[]
    with open(d,'r',encoding='utf-8') as content_file:#abrimos el archivo al que se realizara la prueba
        text=content_file.read()#leemos el archivo completo
    text=text.lower()#pasamos a minusculas
    for p in V:#agregamos a Vd las palabras del vocabulario original que aparescan en el documento de prueba
        cont=text.count(p)
        if cont>0:
            Vd.append(p)
    for c in range(len(C)):#recorremos los dos generos
        if prior[c]==0:#nos aseguramos de que exista al menos 1 documento para cada genero
            score.append(0)
        else:#si existe al menos 1 documento de calcula el logaritmo del numero de documentos del genero sobre el total de numero de documentos
            score.append(log(prior[c]))#lo agregamos a una lista 
        for t in V:#recorremos el vacabulario original
            if t in Vd:#si la palabra del vocabulario original existe en el del documento
                score[c]+=log(contprob[V.index(t)][c])  
            else:#si no existe 
                score[c]+=log(1-contprob[V.index(t)][c])
    return(C[score.index(max(score))]) #regresa el genero donde la prababilidad es mas alta
                
                
val=TrainB(C,D)#llamamos a nuestra funcion de entrenamiento
#print(val[2])
result=applyB(C,val[0],val[1],val[2],d)#llamamos a la funcion de aplicacion de nuestro programa con los valores obtenidos anteriormente en el entrenamiento
print(result)#mostramos resultado