from __future__ import absolute_import, division, print_function, unicode_literals
import random
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn import tree
import numpy as np
from sklearn.model_selection import train_test_split
import csv
import math
import matplotlib.pyplot as plt 
import tensorflow as tf
keras = tf.keras
from tensorflow.keras.models import Model
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
import itertools
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime
from matplotlib.ticker import ScalarFormatter


plt.rc('font', size=20)          # controls default text sizes
plt.rc('axes', titlesize=14)     # fontsize of the axes title
plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=10)    # fontsize of the tick labels
plt.rc('ytick', labelsize=10)    # fontsize of the tick labels
plt.rc('legend', fontsize=10)    # legend fontsize

GeneticScenario='C'
numCF=12
print(datetime.now())
minus=0.016
overprob=0.004
underprob=0.016
control_ln=0
case_ln=0
NF=1000
NSlistlog=[9.2,6.9,6.2,4.6,3.9]
NSlist=[10000,1000,500,100,50]
RCC=0.2
RNF=0.05    

def DatasetgeneratorF(GeneticScenario,RCCad,NS,R,NF,numCF,minus,overprob,underprob,control_ln,case_ln):  
    #First create a dataset of 0s with NS rows and NF columns
    
    one_to_one=0.52381 
    one_to_zero=1-one_to_one
    zero_initially=1-R
    zero_to_one=one_to_zero*R/(1-R) #check this
    zero_to_zero=1-zero_to_one
    cf=[]
    chainlength=NF/(numCF+1)
    for f in range (0,numCF):
        cf.append(int((1+f)*chainlength))
    dt1cases=[]
    dt1controls=[]
    Ydata=[]
    for n in range (0,NS):
        Ydata.append(0) 
    for n in range (0,int(NS*RCCad)):
        Ydata[n]=1    
    numcases=0
    numcontrols=0
            
    while numcases<int(NS*RCCad) or numcontrols<int(NS-NS*RCCad):
        dt1row=[]
        x=random.random()
        if x<zero_initially:
            laststate=0.0
        if x>=zero_initially:
            laststate=1.0
        dt1row.append(laststate)
        for j in range(0,NF-1):
            x=random.random()
            if laststate==0:
                if x<zero_to_one:
                    currentstate=1.0
                else:
                    currentstate=0.0
            if laststate==1:
                if x<one_to_one:
                    currentstate=1.0
                else:
                    currentstate=0.0
            dt1row.append(currentstate)
            laststate=currentstate
        
        for g in range(0,NF):
            if dt1row[g]==0:
                dt1row[g]=2               
            if dt1row[g]==1:
                x=random.random()
                if x<0.404:
                    nv=1
                if x>=0.404 and x<0.986:
                    nv=3
                if x>=0.986:
                    nv=4                    
                startpoint=g
                fv=1
                while fv==1:
                    dt1row[startpoint]=nv
                    startpoint=startpoint+1
                    if startpoint>0 and startpoint<NF:
                        fv=dt1row[startpoint]
                    else:
                        fv=2
                startpoint=g
                fv=1
                while fv==1:
                    dt1row[startpoint]=nv
                    startpoint=startpoint-1
                    if startpoint>0 and startpoint<NF:
                        fv=dt1row[startpoint]
                    else:
                        fv=2
         
        cfrow=[]
        cfbinrow=[]
        for b in range(0,numCF):
            cfrow.append(dt1row[cf[b]])
            if dt1row[cf[b]]==2:
                cfbinrow.append(0)
            if dt1row[cf[b]]!=2:
                cfbinrow.append(1)
        Yval=0
        if GeneticScenario=='A':
            score1=0
            for j in range(0,int(numCF)):
                score1=score1+cfbinrow[j]
            if score1>0:
                Yval=1
        if GeneticScenario=='B':
            score1=0
            for j in range(0,int(numCF/2)):
                score1=score1+cfbinrow[j]*cfbinrow[j+int(numCF/2)]
            if score1>0:
                Yval=1
        if GeneticScenario=='C':
            score1=0
            for j in range(0,int(numCF/2)):
                score1=score1+cfbinrow[j]
            score2=0
            for j in range(int(numCF/2),numCF):
                score2=score2+cfbinrow[j]
            if score1>0 and score2>0:
                Yval=1
        if GeneticScenario=='E':
            score1=0
            for j in range(0,int(numCF)):
                score1=score1+cfbinrow[j]
            score2=0
            for j in range(0,int(numCF/2)):
                score2=score2+cfbinrow[j]*cfbinrow[j+int(numCF/2)]
            if score1>=3:
                Yval=1
            if score2>0:
                Yval=1
        if GeneticScenario=='D':
            score1=0
            for j in range(0,int(numCF/2)):
                if cfrow[j]<2:
                    score1=score1+1
            for j in range(int(numCF/2),numCF):
                if cfrow[j]>2:
                    score1=score1+1       
            if score1>0:
                Yval=1

        x=random.random()
        if Yval==0:
            finalYval=0
            if x<control_ln:
                finalYval=1
        if Yval==1:
            finalYval=1
            if x<case_ln:
                finalYval=0

        for g in range(0,NF):
            if dt1row[g]==2:
                dt1row[g]=-2 

        #if final label is positive add one to positive count, if final label is negative, add one to negative count
        if finalYval==0:
            numcontrols+=1
            if numcontrols<=int(NS-NS*RCCad):
                dt1controls.append(dt1row)
        if finalYval==1:
            numcases+=1
            if numcases<=int(NS*RCCad):
                dt1cases.append(dt1row)

    dt1controls=np.array(dt1controls)
    dt1cases=np.array(dt1cases)
    Xdt1=np.concatenate((dt1cases,dt1controls))
    return Xdt1, Ydata

num_iter=5
resultsarray=[]
    
for NS in NSlist:
    DT_tot=[]
    LR_tot=[]
    NN_tot=[]
    LASSO_tot=[]
    RF_tot=[]
    NN_totnoreg=[]
    rand_tot=[]
              
    numiter=num_iter
    if NS<501:
        numiter=10
    if NS<101:
        numiter=20
    
    for j in range (0,numiter):
        Xdata,Ydata=DatasetgeneratorF(GeneticScenario,RCC,NS,RNF,NF,numCF,minus,overprob,underprob,control_ln,case_ln)
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xdata, Ydata, test_size=0.2, stratify=Ydata)
        Xtrain=np.array(Xtrain)
        Xntrain = (Xtrain - Xtrain.mean()) / Xtrain.std()
        Xtest=np.array(Xtest)
        Xntest = (Xtest - Xtest.mean()) / Xtest.std()

        tuned_parameters = [{ "C": [0.1,1, 10, 100, 1000]}]
        model = GridSearchCV(LogisticRegression(penalty='l1', solver='liblinear',class_weight='balanced'), tuned_parameters,scoring='roc_auc', refit=True)
        train_history=model.fit(Xntrain, Ytrain)
        p=model.predict(Xntest)
        lassoauc_score=roc_auc_score(Ytest,p)
        LASSO_tot.append(lassoauc_score)
        
        tuned_parameters = [{ "ccp_alpha": [0.0, 0.1,1, 10, 100, 1000]}]
        model = GridSearchCV(RandomForestClassifier(class_weight='balanced',max_features=None), tuned_parameters, scoring='roc_auc')
        train_history=model.fit(Xntrain, Ytrain)
        p=model.predict(Xntest)
        rfauc_score=roc_auc_score(Ytest,p)
        RF_tot.append(rfauc_score)

        tuned_parameters = [{ "ccp_alpha": [0.0, 0.1,1, 10, 100, 1000]}]
        model = GridSearchCV(tree.DecisionTreeClassifier(class_weight='balanced'), tuned_parameters, scoring='roc_auc')
        train_history=model.fit(Xntrain, Ytrain)
        p=model.predict(Xntest)
        treeauc_score=roc_auc_score(Ytest,p)
        DT_tot.append(treeauc_score)
        
        tuned_parameters = [{ "C": [0.1,1, 10, 100, 1000]}]
        model = GridSearchCV(LogisticRegression(C=1,class_weight='balanced'), tuned_parameters,scoring='roc_auc', refit=True)
        train_history=model.fit(Xntrain, Ytrain)
        p=model.predict(Xntest)
        LRauc_score=roc_auc_score(Ytest,p)
        LR_tot.append(LRauc_score)

        YtrainNN = tf.keras.utils.to_categorical(Ytrain, 2)
        YtestNN = tf.keras.utils.to_categorical(Ytest, 2)
        base_learning_rate = 0.01
        es = EarlyStopping(monitor='val_auc', mode='max', verbose=0, patience=5)
        model = tf.keras.Sequential()
        model.add(keras.layers.Dense(NF/2, activation="relu", input_shape=(NF,),kernel_regularizer=tf.keras.regularizers.L1(0.001)))
        prediction_layer = keras.layers.Dense(2, activation='softmax')
        model.add(prediction_layer)
        model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate), loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),metrics=['AUC'])
        train_history=model.fit(Xntrain, YtrainNN,validation_split=0.1,epochs=500,verbose=1, class_weight={0: 1.0, 1:RCC},callbacks=[es])
        p=model.predict(Xntest)
        NNauc_score=roc_auc_score(YtestNN,p)
        NN_tot.append(NNauc_score)
        print(NNauc_score)
        
        model = tf.keras.Sequential()
        model.add(keras.layers.Dense(NF/2, activation="relu", input_shape=(NF,)))
        prediction_layer = keras.layers.Dense(2, activation='softmax')
        model.add(prediction_layer)
        model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate), loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),metrics=['AUC'])
        train_history=model.fit(Xntrain, YtrainNN,validation_split=0.1,epochs=500,verbose=1, class_weight={0: 1.0, 1:RCC},callbacks=[es])
        p=model.predict(Xntest)
        NNauc_score=roc_auc_score(YtestNN,p)
        NN_totnoreg.append(NNauc_score)
        
        y_val_cat_prob=[]
        for ii in range (0,len(Ytest)):
            x=random.random()
            if x>0.2:
                y_val_cat_prob.append(0)
            if x<=0.2:
                y_val_cat_prob.append(1)         
        randauc_score=roc_auc_score(Ytest,y_val_cat_prob)
        rand_tot.append(randauc_score)  
        
    resultsarray.append([NF,NS,RCC,numiter,sum(LR_tot)/len(LR_tot),(np.std(LR_tot))/math.sqrt(len(LR_tot)),sum(DT_tot)/len(DT_tot),(np.std(DT_tot))/math.sqrt(len(DT_tot)),sum(NN_totnoreg)/len(NN_totnoreg),(np.std(NN_totnoreg))/math.sqrt(len(NN_totnoreg)),sum(RF_tot)/len(RF_tot),(np.std(RF_tot))/math.sqrt(len(RF_tot)),sum(NN_tot)/len(NN_tot),(np.std(NN_tot))/math.sqrt(len(NN_tot)),sum(LASSO_tot)/len(LASSO_tot),(np.std(LASSO_tot))/math.sqrt(len(LASSO_tot)),sum(rand_tot)/len(rand_tot),(np.std(rand_tot))/math.sqrt(len(rand_tot))])    

resultsarraytitle=[['NF','NS','RCC','number iterations','LR_ROC_AUC','LR_err','DT_ROC_AUC','DT_err','NNnoreg AUC','NN_err','RF AUC','RF_err','NN AUC','NN_err','LR AUC','LRnoreg_err','rand','rand_err']]
forcsv=resultsarraytitle+resultsarray
with open('results/cnv Model performance by dataset size scenario '+str(GeneticScenario)+str(numCF)+'.csv','w', newline='') as result_file:
    wr = csv.writer(result_file, dialect='excel')
    wr.writerows(forcsv)  


plt.clf
resultsarray=np.array(resultsarray)
lrnf=len(NSlist)
i=0
ax = plt.gca()
ax.set_ylim([0.4, 1.05])

plt.errorbar(NSlistlog,resultsarray[0:len(NSlist),4],yerr=resultsarray[0:len(NSlist),5],color='red', markerfacecolor='red')
plt.errorbar(NSlistlog,resultsarray[0:len(NSlist),14],yerr=resultsarray[0:len(NSlist),15],color='red', linestyle='dashed', markerfacecolor='red')
plt.errorbar(NSlistlog,resultsarray[0:len(NSlist),8],yerr=resultsarray[0:len(NSlist),9],color='green', markerfacecolor='green')
plt.errorbar(NSlistlog,resultsarray[0:len(NSlist),12],yerr=resultsarray[0:len(NSlist),13],color='green', linestyle='dashed', markerfacecolor='green')
plt.errorbar(NSlistlog,resultsarray[0:len(NSlist),6],yerr=resultsarray[0:len(NSlist),7],color='blue',markerfacecolor='blue')
plt.errorbar(NSlistlog,resultsarray[0:len(NSlist),10],yerr=resultsarray[0:len(NSlist),11],color='orange',markerfacecolor='orange')
plt.errorbar(NSlistlog,resultsarray[0:len(NSlist),16],yerr=resultsarray[0:len(NSlist),17],color='black', markerfacecolor='black')

ax.xaxis.set_major_formatter(ScalarFormatter())
plt.xticks(ticks=NSlistlog, labels=['10000','1000','500','100','50'])
plt.legend(['LR','LR+L1','NN','NN+L1','DT','RF','random'])
plt.xlabel('Number of samples') 
plt.ylabel('Test set ROC AUC') 
plt.title('Model performance by dataset size (scenario '+str(GeneticScenario)+str(numCF)+')')
plt.savefig('results/cnv Model performance by dataset size (scenario '+str(GeneticScenario)+str(numCF)+').jpg')
plt.clf()
plt.cla()
plt.close()

plt.clf
