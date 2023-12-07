2# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 18:15:08 2022

@author: eg632
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
            if count==2:
                channels=["EEG.AF3","EEG.F7","EEG.F3","EEG.FC5","EEG.T7","EEG.P7","EEG.O1","EEG.O2","EEG.P8","EEG.T8","EEG.FC6","EEG.F4","EEG.F8","EEG.AF4"]
                line=line.strip().split(",")
                ind=[]
                for i in range(len(line)):
                    if line[i] in channels:
                        ind.append(i)
            if count>2:
                if count!=1:
                    line=line.strip().split(",")
                #line=np.array(line)
                final=[]
                
                for i in ind:
                    final.append(line[i])
                data.append(final)
                if count>10002:
                    break
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
    return(train_ind, test_ind, Y_train, Y_test)
        
        

def format_data(data):
    new_data=[]
    for i in data:
        new_data.append(i[-2])
    new_data.append(data[1][-1])
    return new_data

def bin_data(data):
    bins=[]
    for i in data:
        if i >= 5 and i < 6:
            bins.append(0)
        elif i>=6 and i<7:
            bins.append(1)
        elif i>=7 and i<8:
            bins.append(2)
        elif i>=8 and i<9:
            bins.append(3)
        elif i>=9 and i<10:
            bins.append(4)
    return bins
big_data=[]
for i in range(1,32):
    if(i!=26):
        data=[]
        filename="Trial "+str(i)+"_EPOCX.csv"
        data=read_in_file(filename, data)
        #print(np.shape(data))
        #print(np.ndim(data))
        #data=format_data(data)
        if(i!=16):
            big_data.append(np.array(data))

#print("CHECK") 
X=[]
Y=[8,5,8,8,6.5,6.5,8,9,7.5,9.5,7.5,7.5,9,7,7.5,6.5,9,9,6.5,8,7,9,8,9.5,9.5,8.2,8.5,7.75,7]
Y=bin_data(Y)

#X=np.array([big_data[0][:-1]])
#Y=[big_data[-1]]
#big_data.remove(big_data[0])
#print(big_data)
#print(data[0][:-1])
for i in big_data:
    #X=np.append(X,(np.array([i[:-1]])), axis=0)
    X.append(i)
X=np.array([X])
Y=np.array(Y)
#print(X[0][0])


#print(len(Y))
#print(len(X))
x_train, x_test, y_train, y_test = train_test_split(X[0], Y, test_size = 0.3, random_state = 1)  
#print(np.shape(x_train))
#print(y_train)
#print(np.shape(y_train))
#print(y_train[0])

print(x_train[0])
    

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(10000,14)),
  tf.keras.layers.Dense(1000, activation='relu'),
  tf.keras.layers.Dropout(0.4),
  tf.keras.layers.Dense(5, activation='softmax')
])



model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
boo=True
for i in x_train:
#    print(np.shape(i))
    for j in i:
        for k in range(len(j)):
            j[k]=float(j[k])
x_test2=[]
l=len(x_train)
for i in x_train:
    x_testh=[]
#    print(np.shape(i))
    for j in i:
        for k in range(len(j)):
            x_testh.append(float(j[k]))
    x_testh=np.array(x_testh)
    np.reshape(x_testh,(10000,14))
    x_test2.append(x_testh)
#print(len(x_test2))
x_test2=np.array(x_test2)
x_train=x_test2
x_train=np.reshape(x_train,(l,10000,14))
##for i in x_train:
##    for j in i:
##        print(type(j))

x_test2=[]
l=len(x_test)
#print(l)
for i in x_test:
    x_testh=[]
#    print(np.shape(i))
    for j in i:
        for k in range(len(j)):
            x_testh.append(float(j[k]))
    x_testh=np.array(x_testh)
    np.reshape(x_testh,(10000,14))
    x_test2.append(x_testh)
#print(len(x_test2))
x_test2=np.array(x_test2)
x_test=x_test2
x_test=np.reshape(x_test,(l,10000,14))
#print(np.shape(x_test))
#for i in x_train:
##    print(np.shape(i))
#    for j in i:
#        for k in range(len(j)):
#            print(type(j[k]))
#np.asarray(y_train,type(x_train[0][0]))    
y_train2=[]
for i in range(len(y_train)):
#    #y_train[i]=float(y_train[i])
    y_train2.append(float(y_train[i]))

y_train2=np.array(y_train2)

y_train=y_train2
#print(np.shape(x_train))

#for i in x_test:
#    for j in range(len(i)):
#        i[j]=float(i[j])
y_test2=[]
for i in range(len(y_test)):
    y_test2.append(float(y_test[i]))

y_test2=np.array(y_test2)
y_test=y_test2
print(np.shape(x_train))
model.fit(x_train, y_train, epochs=10   )
print(type(x_test[0][0]))
model.evaluate(x_test, y_test)

filename="Trial "+str(16)+"_EPOCX.csv"
data=read_in_file(filename, data)
x_test2=[]

for i in data:
    x_testh=[]
#    print(np.shape(i))
    for j in range(len(i)):
        x_testh.append(float(i[j]))
    x_testh=np.array(x_testh)
    x_test2.append(x_testh)
#x_test2=np.array(x_test2)
#x_test2=np.reshape(x_test2,(1,10000,14))
#y_test2=[7]
x_test2=x_test[0]
x_test2=np.reshape(x_test2,(1,10000,14))
y_test2=[y_test[0]]
#y_test2=[3]
y_test2=np.array(y_test2)
print(type(x_test2[0][0]))
#model.evaluate(x_test2, y_test2)