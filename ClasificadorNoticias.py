import string
import re
import nltk
import io
import joblib
import pickle

from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from tkinter import *

import xlsxwriter

from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, make_scorer, recall_score, accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from nltk.stem import PorterStemmer
from pandas import DataFrame
import pandas as pd
import numpy as np
import os

#COMMIT DE PRUEBA

    #quitar esta linea y organizar todo
vectorizer = TfidfVectorizer()

def cargarFicheros(cadena):
    print("Cargando rutas de los ficheros...")
    textos = []
    for text in list(os.scandir(cadena)):
        # Escanear todo el directorio
        print(text.path)
        textos.append(text.path)  # Agregar solo el path de los archivos de texto(archivos.txt)
    textos.sort() # Ordenar todas las entradas, de path de archivos de todos los textos en el directorio
    print("FINALIZADO.")
    return textos

#crea una lista con los valores segun la noticia 1 Despoblacion y 0 NO DESPOBLACION
def cargarClasificador(listaNoticia, clase):
    lista = []
    if(clase == 1):
        for _ in range(len(listaNoticia)):
            lista.append(1)
    else:
        for _ in range(len(listaNoticia)):
            lista.append(0)
    return lista

#pasamos las rutas a texto
def cargarNoticias(listaPath):
    print("Cargando noticias...")
    listaTextos = []
    for x in listaPath:
        #el encoding es importante para solucionar el error UnicodeDecodeError: 'charmap' codec can't               decode byte 0x9d
        archivo = open(x, encoding='utf-8')
        listaTextos.append(archivo.read())
        archivo.close()
    print("Terminado.")
    return listaTextos

porter=PorterStemmer()

def stemSentence(sentence):
    token_words=word_tokenize(sentence)
    token_words
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)

def procesar(texto):
    texto = re.sub('\w*\d\w*', '', texto) # metodo para borrar todos los numeros
    texto = texto.lower() # ponemos el texto todo en minusculas
    texto = re.sub('[%s]' % re.escape(string.punctuation), '', texto) # quitamos la puntuacion del texto
    texto = re.sub('[’‘“”»«¿?]', '', texto)
    texto = re.sub('\s(https.*?)\s', '', texto)

    misPalabras = word_tokenize(texto) # hago una lista de palabras
    filterWordList = misPalabras[:]

    for word in misPalabras:
        if word in stopwords.words('Spanish'):#si la palabra esta en las lista de stopWords la borra
            filterWordList.remove(word)

    lemmatizer = WordNetLemmatizer()#lematizamos

    for x in range(len(filterWordList)):
        filterWordList[x] = lemmatizer.lemmatize(filterWordList[x])#aprovechando que ya les quitamos los stop wors
        filterWordList[x] = stemSentence(filterWordList[x])
    
    cadena = " ".join(filterWordList)#convierto la lista con las palabras stop words eliminadas en cadena para tagearlo
    return cadena

def lemTokProces(listaTextos):
    print("Procesando el texto...")
    listaTokLem =[]
    for x in range(len(listaTextos)):
        listaTokLem.append(procesar(listaTextos[x]))
    print("Textos procesados con exito.")
    return listaTokLem

def AnaiveBayes(x_train, y_train):
    model = LinearRegression()
    acc = cross_val_score(estimator= model, X = x_train, y = y_train, cv = 2)#presicion
    modeloEntrenado = model.fit(x_train, y_train)
    print(acc)
    print("presicion train: %0.3f"% acc.mean())
    #matrizConfucionTrain = confusion_matrix(y_train, prediccion)
    #print("Matriz de confucion")
    #print(matrizConfucionTrain)
    guardarModelo(modeloEntrenado)

def AdecisionTree(x_train, y_train):
    model = DecisionTreeClassifier(random_state=1)
    #parametros = {'criterion' : ['gini', 'entropy'], 'splitter' : ['best', 'random']}
    acc = cross_val_score(estimator= model, X = x_train, y = y_train, cv = 2)#presicion
    modeloEntrenado = model.fit(x_train, y_train)
    print(acc)
    print("presicion train: %0.3f"% acc.mean())
    #matrizConfucionTrain = confusion_matrix(y_train)
    #print("Matriz de confucion")
    #print(matrizConfucionTrain)
    guardarModelo(modeloEntrenado)

def AKnn(x_train, y_train):
    model = KNeighborsClassifier(n_neighbors = 3)
    acc = cross_val_score(estimator= model, X = x_train, y = y_train, cv = 2)#presicion
    modeloEntrenado = model.fit(x_train, y_train)
    print(acc)
    print("presicion train: %0.3f"% acc.mean())
    #matrizConfucionTrain = confusion_matrix(y_train, prediccion)
    #print("Matriz de confucion")
    #print(matrizConfucionTrain)
    guardarModelo(modeloEntrenado)

def guardarModelo(modelo):
    nombreArchivo = txtGmodelo.get()
    print(nombreArchivo)
    filename = nombreArchivo + '.sav'
    print(filename)
    pickle.dump(modelo, open(filename, 'wb'))
    print("MODELO GUARDADO CON EXITO.")

def cargarModelo(pathModelo):
    # load the model from disk
    loaded_model = joblib.load(pathModelo)
    #result = loaded_model.score(X_test, Y_test)
    return loaded_model

def contarnoticasD(noticiasDespoblacion):
    numeroND = len(noticiasDespoblacion)
    return numeroND

def main(rutaDes, rutaNoDes, algoritmo):
    listaClasficar = []
    listaPath = cargarFicheros(rutaDes)

    listaClasficar = cargarClasificador(listaPath, 1)

    listaPath2 = []
    listaPath2 = cargarFicheros(rutaNoDes)
    
    contarNoticasDespoblacion = contarnoticasD(listaPath)
    contarNoticiasNoDespoblacion = contarnoticasD(listaPath2)

    #terminamos de cargar el clasificardor con el resto de noticias
    listaClasficar = listaClasficar + cargarClasificador(listaPath2, 0)

    #jutamos las 2 listas de despoblacion y no despoblacion
    listaPathTotal = listaPath + listaPath2
    listaNoticias = []
    #cargamos los textos de las noticias
    listaNoticias = cargarNoticias(listaPathTotal)

    lsTextosLemTok = []
    lsTextosLemTok = lemTokProces(listaNoticias)

    vectors = vectorizer.fit_transform(lsTextosLemTok)# matriz ftf
    
    if algoritmo == 'regresion lineal':
        print("iniciando algoritmo regrasion lineal.")
        AnaiveBayes(vectors, listaClasficar)
    elif algoritmo == 'decision tree':
        print("iniciando algoritmo decision tree.")
        AdecisionTree(vectors, listaClasficar)
    elif algoritmo == 'knn':
        print("iniciando algoritmo knn.")
        AKnn(vectors, listaClasficar)

    imprimirporpantalla(contarNoticasDespoblacion, contarNoticiasNoDespoblacion)

def main2(modeloCargado, pathDesconocido):
    print("iniciando testeo...")
    listaPath = []
    listaPath = cargarFicheros(pathDesconocido)
    listaNoticias = []
    #cargamos los textos de las noticias
    listaNoticias = cargarNoticias(listaPath)
    lsTextosLemTok = []
    lsTextosLemTok = lemTokProces(listaNoticias)

    vectors = vectorizer.transform(lsTextosLemTok)#matriz x test

    modelo = cargarModelo(modeloCargado)
    prediccion = modelo.predict(vectors)
    print("Prediccion del test.", prediccion)
    print("test finalizado con exito.")
    print("Enseñando nombre de los archivos y su categoria.")

    nnDesco = len(listaPath)
    libro = xlsxwriter.Workbook('Resultado_Algoritmo_Noticias.xlsx')
    hoja = libro.add_worksheet()
    row=0
    col=0
    for i in range(len(prediccion)):
        categoria=""
        if(prediccion[i] == 1):
            categoria=" DESPOBLACION."
            print(listaPath[i], categoria)
            hoja.write(row, col, listaPath[i])
            rutanoticia = listaPath[i]
            hoja.write(row, col + 1, categoria)
            row += 1
        else:
            categoria = " NO DESPOBLACION"
            print(listaPath[i], categoria)
            hoja.write(row, col, listaPath[i])
            hoja.write(row, col + 1, categoria)
            row += 1
        imprimirporpantalla2(rutanoticia, categoria, nnDesco)
    libro.close()
#####################################################################
def ejecutar():
    print(txtGmodelo.get())
    print(txtNd1.get(), txtNdn1.get(), combo.get())
    main(txtNd1.get(), txtNdn1.get(), combo.get())

def ejecutar2():
    print(txtNd2.get(), txtNdn2.get())
    main2(txtNdn2.get(), txtNd2.get())

def conseguirRutaDespoblacion():
    root = Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory()
    print(folder_selected)
    txtNd1.insert(0, folder_selected)
    print(txtNd1.get())

def conseguirRutaNoDespoblacion():
    root = Tk()
    root.withdraw()
    folder_selected_No_despoblacion = filedialog.askdirectory()
    print(folder_selected_No_despoblacion)
    txtNdn1.insert(0, folder_selected_No_despoblacion)
    print(txtNdn1.get())

def conseguirRutaNotDesconocidas():
    root = Tk()
    root.withdraw()
    folder_selected_Not_desconocidas = filedialog.askdirectory()
    print(folder_selected_Not_desconocidas)
    txtNd2.insert(0, folder_selected_Not_desconocidas)
    print(txtNd2.get())

def conseguirRutaModelo():
    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    folder_selected_Modelo = askopenfilename() # show an "Open" dialog box and return the path to the selected file
    print(folder_selected_Modelo)
    txtNdn2.insert(0, folder_selected_Modelo)
    print(txtNdn2.get())

def imprimirporpantalla(ND, NND):
    listbox_tk1.insert(END, ("VISTA PREVIA"))
    listbox_tk1.insert(END, ())
    listbox_tk1.insert(END,("      Ejemplares Despoblacion: ", ND))
    listbox_tk1.insert(END,("      Ejemplares No Despoblacion: ", NND))
    listbox_tk1.insert(END,("      TOTAL Ejemplares: ", ND + NND))
    listbox_tk1.insert(END,("      Algoritmo seleccionado: ", combo.get()))

def imprimirporpantalla2(rutanoticia, categoria, nnDesco):
    print("*****ENTRO AL METODO*****")
    listbox_tkk1.insert(END, ("      NOTICAS     /    CATEGORIA"))
    listbox_tkk1.insert(END, ())
    listbox_tkk1.insert(END,(rutanoticia,"   /   ",categoria))

from tkinter import *
from tkinter import ttk
from tkinter.ttk import *
from tkinter import scrolledtext

#Se crea la ventana sobre la que vamos a trabajar
ventana = Tk()
ventana.title("Clasificador")
ventana.geometry("850x600+250+75")
ventana.resizable(width=False, height=False)

tab_control = ttk.Notebook(ventana)

tab1 = ttk.Frame(tab_control)
tab2 = ttk.Frame(tab_control)

tab_control.add(tab1, text='Entrenamiento')

lblNd1 = Label(tab1, text= 'Noticias de despoblación:')
txtNd1 = Entry(tab1,width=40)
lblNd1.place(x=50, y=43)
txtNd1.place(x=215, y=43)
btnNd1 = Button(tab1, text="Abrir", command = conseguirRutaDespoblacion)
btnNd1.place(x=480, y=40)

lblNdn1 = Label(tab1, text= 'Noticias de no despoblación:')
txtNdn1 = Entry(tab1,width=40)
lblNdn1.place(x=50, y=82)
txtNdn1.place(x=215, y=82)
btnNdn1 = Button(tab1, text="Abrir", command = conseguirRutaNoDespoblacion)
btnNdn1.place(x=480, y=79)

lblSa = Label(tab1, text= 'Seleccionar algoritmo:')
lblSa.place(x=50,y=125)

combo = Combobox(tab1, state="readonly")
combo['values']= ('regresion lineal', "decision tree", 'knn')
combo.current(0)
combo.place(x=200,y=125)

listbox_tk1 = Listbox(tab1,width=60,height=7)
listbox_tk1.place(x=100,y=180)

btnEjec = Button(tab1, text="Ejecutar", command = ejecutar)
btnEjec.place(x=570,y=213)

listbox_tk2 = Listbox(tab1,width=100,height=8)
listbox_tk2.place(x=100,y=320)

lblG = Label(tab1, text= 'Nombre del clasificador:')
txtGmodelo = Entry(tab1,width=55)
#btnG = Button(tab1, text="Guardar")
lblG.place(x=40, y=500)
txtGmodelo.place(x=190, y=498)
#btnG.place(x=550, y=500)

#---------------------------------------------------------
#Tab2
tab_control.add(tab2, text='Clasificación')
tab_control.pack(expand=1, fill='both')

lblNd2 = Label(tab2, text= 'Noticias Desconocidas: ')
txtNd2 = Entry(tab2,width=40)
lblNd2.place(x=50, y=43)
txtNd2.place(x=215, y=43)
btnNd2 = Button(tab2, text="Abrir", command = conseguirRutaNotDesconocidas)
btnNd2.place(x=480, y=40)

lblNdn2 = Label(tab2, text= 'Modelo de Clasificador:')
txtNdn2 = Entry(tab2,width=40)
btnNdn2 = Button(tab2, text="EJECUTAR", command=ejecutar2)
lblNdn2.place(x=50, y=82)
txtNdn2.place(x=215, y=82)
btnNdn2.place(x=300, y=130)

btnNdn3 = Button(tab2, text="Abrir", command = conseguirRutaModelo)
btnNdn3.place(x=480, y=79)

listbox_tkk1 = Listbox(tab2, width=70,height=10)
listbox_tkk1.place(x=100,y=180)

listbox_tkk2 = Listbox(tab2,width=30,height=10)
listbox_tkk2.place(x=550,y=180)

ventana.mainloop()