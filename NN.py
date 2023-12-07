# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 15:06:46 2021

@author: 1501309
"""

import csv
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
import numpy as np
import random
def read_in_file(filename, data):
    with open(filename) as f:
        i=0
        count=0
        for line in f:
            count+=1
            if count==1:
                channels=["EEG.AF3","EEG.F7","EEG.F3","EEG.FC5","EEG.T7","EEG.P7","EEG.O1","EEG.O2","EEG.P8","EEG.T8","EEG.FC6","EEG.F4","EEG.F8","EEG.AF4"]
                line=line.strip().split(",")
                ind=[]
                for i in range(len(line)):
                    if line[i] in channels:
                        ind.append(i)
            line=line.strip().split(",")
            line=line[1:6]
            final=[]
            for i in ind:
                final.append(line[i])
            data.append(final)
    data.remove(data[0])
    return data

def split_train_test(size, deci):
    test_ind=[]
    train_ind=[]
    Y=[8,5,8,8,6.5,6.5,8,9,7.5,9.5,7.5,7.5,9,7,7.5,6.5,9]
    for i in range(size):
        train_ind.append(i)
    for i in range(deci*size):
        r=int(random.random()*size)
        while r not in train_ind:
            r=int(random.random()*size)
        train_ind.remove(r)
    Y_train=[]
    Y_test=[]
    for i in train_ind:
        Y_train.append(Y[i])
    for i in test_ind:
        Y_test.append(Y[i])
    return(train_ind, test_ind)
        
        

def format_data(data):
    new_data=[]
    for i in data:
        if(type(i[-2])!=type(float(1231.1324))):
            print("a;sdhfakjsdhflkjasdh")
        new_data.append(i[-2])
    new_data.append(data[1][-1])
    return new_data

big_data=[]
for i in range(1,19):
    data=[]
    filename="Trial"+str(i)+"EPOCX.csv"
    data=read_in_file(filename, data)
    data=format_data(data)
    big_data.append(np.array(data))
#print("CHECK") 
X=[]
Y=[]
#X=np.array([big_data[0][:-1]])
#Y=[big_data[-1]]
#big_data.remove(big_data[0])
#print(big_data)
#print(data[0][:-1])
for i in big_data:
    #X=np.append(X,(np.array([i[:-1]])), axis=0)
    X.append(i[:-1])
    Y.append(i[-1])
X=np.array([X])
Y=np.array(Y)
#print(X[0][0])

for i in range(len(Y)):
    if Y[i]=="a":
        Y[i]=1
    else:
        Y[i]=0
x_train, x_test, y_train, y_test = train_test_split(X[0], Y, test_size = 0.3, random_state = 1)  
#print(y_train[0])

       

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(11000,1)),
  tf.keras.layers.Dense(1000, activation='relu'),
  tf.keras.layers.Dropout(0.4),
  tf.keras.layers.Dense(2, activation='softmax')
])

for i in x_train:
    for j in i:
        if type(i)==type("."):
            print("pulling my hair out part 2")

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
boo=True
print("CHECK 2")
for i in x_train:
    for j in range(len(i)):
        i[j]=float(i[j])
#for i in x_train:
#    for j in i:
#        print(type(j))
        

np.asarray(y_train,type(x_train[0][0]))    
y_train2=[]
for i in range(len(y_train)):
    #y_train[i]=float(y_train[i])
    y_train2.append(float(y_train[i]))

y_train2=np.array(y_train2)
#for i in y_train2:
#    print(type(i))
y_train=y_train2
print(np.shape(x_train))

for i in x_test:
    for j in range(len(i)):
        i[j]=float(i[j])
y_test2=[]
for i in range(len(y_test)):
    y_test2.append(float(y_test[i]))

y_test2=np.array(y_test2)
y_test=y_test2
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)