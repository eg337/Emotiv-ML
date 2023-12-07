# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 15:14:50 2021

@author: 1501309
"""
import csv
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
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

def format_data(data):
    new_data=[]
    for i in data:
        new_data.append(i[-2])
    #print(data[6])
    new_data.append(data[1][-1])
    return new_data
def bin_data(data):
    bins=[]
    for i in data:
        if i >= 5 and i < 6:
            bins.append(1)
        elif i>=6 and i<7:
            bins.append(2)
        elif i>=7 and i<8:
            bins.append(3)
        elif i>=8 and i<9:
            bins.append(4)
        elif i>=9 and i<10:
            bins.append(5)
    return bins
print("CHECK")
big_data=[]
for i in range(1,31):
    if(i!=26):
        data=[]
        filename="Trial "+str(i)+"_EPOCX.csv"
        data=read_in_file(filename, data)
        #print(np.shape(data))
        #print(np.ndim(data))
        #data=format_data(data)
        big_data.append(np.array(data))
print("CHECK") 
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
#print(np.shape(X))

Y=np.array(Y)
#print(X[0][0])


#print(len(Y))
#print(len(X))
print(Y)
x_train, x_test, y_train, y_test = train_test_split(X[0], Y, test_size = 0.3, random_state = 1) 
print(y_train)

clf=RandomForestClassifier(n_estimators=100)
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
    np.reshape(x_testh,(140000))
    x_test2.append(x_testh)
#print(len(x_test2))
x_test2=np.array(x_test2)
x_train=x_test2
x_train=np.reshape(x_train,(l,140000))
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
    np.reshape(x_testh,(140000))
    x_test2.append(x_testh)
#print(len(x_test2))
x_test2=np.array(x_test2)
x_test=x_test2
x_test=np.reshape(x_test,(l,140000))
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
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
#print(y_pred)
#print(y_test)
#print(y_pred)
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


filename="Trial "+str(31)+"_EPOCX.csv"
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
x_test2=np.reshape(x_test2,(1,140000))
y_test2=[7]
#y_test2=[3]
y_test2=np.array(y_test2)
y_pred=clf.predict(x_test2)

