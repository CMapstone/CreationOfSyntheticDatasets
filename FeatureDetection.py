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
import shap
shap.initjs()
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime
from matplotlib.ticker import ScalarFormatter
print(datetime.now())

GeneticScenario='A'
numCF=2
RNF=0.006
minus=0.016
overprob=0.004
underprob=0.016
control_ln=0
case_ln=0
NF=1000
NSlistlog=[9.2,6.9,6.2,4.6,3.9]
NSlist=[10000,1000,500,100,50]
RCC=0.2
num_iter=5

plt.rc('font', size=20)          # controls default text sizes
plt.rc('axes', titlesize=14)     # fontsize of the axes title
plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=10)    # fontsize of the tick labels
plt.rc('ytick', labelsize=10)    # fontsize of the tick labels
plt.rc('legend', fontsize=10)    # legend fontsize

def tothitscalcLRcoefs(model,numCF,cf):                       
    coefs=model.best_estimator_.coef_[0]
    coefs_notsorted=np.absolute(coefs)
    coefs_sorted=np.absolute(coefs)
    coefs_sorted.sort()
    
    tothits=0   
    for isv in coefs_sorted[(0-numCF):]:
        ind=np.where(coefs_notsorted == isv)
        for cf_ in cf:  
            if int(ind[0][0])==cf_:
                if isv>0:
                    tothits=tothits+1
    rank=0  
    found=0
    for ij in range(1,NF+1):
        isv=coefs_sorted[-ij]
        if isv ==0:
            break
        rank=rank+1
        ind=np.where(coefs_notsorted == isv)
        for cf_ in cf:    
            if int(ind[0][0])==cf_:
                found=found+1
                lastrank=rank
                break
        if found>=(numCF):
            break
            
    return tothits,rank,found

def tothitscalc(model,NN,numCF, cf,Xntest):
    print(datetime.now())
    Xntest=np.array(Xntest)
    explainer = shap.Explainer(model.predict,Xntest,verbose=0)
    print(datetime.now())
    shap_values = explainer.shap_values(Xntest)
    print(datetime.now())
    shpv_con=np.array(shap_values) 
    if NN=='Y':
        shpv=np.array(shap_values) 
        shpv_con=shpv[:,:,1]
    shpv_conv2=np.absolute(shpv_con)
    control_shaps=shpv_conv2.mean(axis=0)
    CSsorted=list(set(control_shaps))
    CSsorted.sort()
    tothits=0        
    for isv in CSsorted[(0-numCF):]:
        ind=np.where(control_shaps == isv)
        for cf_ in cf:
            if int(ind[0][0])==cf_:
                if isv>0:
                    tothits=tothits+1
    rank=0  
    found=0
    for ij in range(1,NF+1):
        isv=CSsorted[-ij]
        if isv ==0:
            break
        rank=rank+1
        ind=np.where(control_shaps == isv)
        for cf_ in cf:    
            if int(ind[0][0])==cf_:
                found=found+1
                lastrank=rank
                break
        if found>=(numCF):
            break

    
    return tothits,rank,found

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

cf=[]
chainlength=NF/(numCF+1)
for f in range (0,numCF):
    cf.append(int((1+f)*chainlength))

resultsarray=[]   
resultsarray2=[]
resultsarray3=[]
for NS in NSlist:           
    
    DT_tot=[]
    NN_tot=[]
    LASSO_tot=[]
    RF_tot=[]
    NRNN_tot=[]
    LASSOSHAP_tot=[]
    NRLASSOSHAP_tot=[]
    NRLASSO_tot=[]

    DT_FOUND=[]
    NN_FOUND=[]
    LASSO_FOUND=[]
    RF_FOUND=[]
    NRNN_FOUND=[]
    LASSOSHAP_FOUND=[]
    NRLASSOSHAP_FOUND=[]
    NRLASSO_FOUND=[]

    DT_LR=[]
    NN_LR=[]
    LASSO_LR=[]
    RF_LR=[]
    NRNN_LR=[]
    LASSOSHAP_LR=[]
    NRLASSOSHAP_LR=[]
    NRLASSO_LR=[]
                 
    numiter=num_iter
    if NS<99:
        numiter=10
    for j in range (0,numiter):
        Xdata,Ydata=DatasetgeneratorF(GeneticScenario,RCC,NS,RNF,NF,numCF,minus,overprob,underprob,control_ln,case_ln)
        Ytrain=Ydata
        Xtrain=np.array(Xdata)
        Xntrain = (Xtrain - Xtrain.mean()) / Xtrain.std()
        if NS<101:
            Xntest = (Xtrain - Xtrain.mean()) / Xtrain.std()    
        if NS>100:
            splitvalue=100/NS
            Xtrain, Xtest, Ytra, Yte = train_test_split(Xdata, Ydata, test_size=splitvalue, stratify=Ydata)
            Xntest = (Xtest - Xtrain.mean()) / Xtrain.std()

        tuned_parameters = [{ "C": [0.1,1, 10, 100, 1000]}]
        model = GridSearchCV(LogisticRegression(penalty='l1', solver='liblinear',class_weight='balanced'), tuned_parameters,scoring='roc_auc', refit=True)
        train_history=model.fit(Xntrain, Ytrain) 
        tothits,rank,found=tothitscalcLRcoefs(model,numCF,cf)
        LASSO_tot.append(tothits/numCF)
        LASSO_LR.append(rank)
        LASSO_FOUND.append(found)
        tothits,rank,found=tothitscalc(model,'N',numCF, cf,Xntest)
        LASSOSHAP_tot.append(tothits/numCF)
        LASSOSHAP_LR.append(rank)
        LASSOSHAP_FOUND.append(found)
        
        model = GridSearchCV(LogisticRegression(C=1,class_weight='balanced'), tuned_parameters,scoring='roc_auc', refit=True)
        train_history=model.fit(Xntrain, Ytrain)     
        tothits,rank,found=tothitscalcLRcoefs(model,numCF,cf)
        NRLASSO_tot.append(tothits/numCF)
        NRLASSO_LR.append(rank)
        NRLASSO_FOUND.append(found)
        tothits,rank,found=tothitscalc(model,'N',numCF, cf,Xntest)
        NRLASSOSHAP_tot.append(tothits/numCF)
        NRLASSOSHAP_LR.append(rank)
        NRLASSOSHAP_FOUND.append(found)

        YtrainNN = tf.keras.utils.to_categorical(Ytrain, 2)
        es = EarlyStopping(monitor='val_auc', mode='max', verbose=0, patience=5)
        base_learning_rate = 0.01
        model = tf.keras.Sequential()
        model.add(keras.layers.Dense(NF/2, activation="relu", input_shape=(NF,),kernel_regularizer=tf.keras.regularizers.L1(0.001)))
        prediction_layer = keras.layers.Dense(2, activation='softmax')
        model.add(prediction_layer)
        model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate), loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),metrics=['AUC'])
        train_history=model.fit(Xntrain, YtrainNN,validation_split=0.1,epochs=500,verbose=1, class_weight={0: 1.0, 1:RCC},callbacks=[es])
        tothits,rank,found=tothitscalc(model,'Y',numCF, cf,Xntest)
        NN_tot.append(tothits/numCF)
        NN_LR.append(rank)
        NN_FOUND.append(found)
    
        model = tf.keras.Sequential()
        model.add(keras.layers.Dense(NF/2, activation="relu", input_shape=(NF,)))
        prediction_layer = keras.layers.Dense(2, activation='softmax')
        model.add(prediction_layer)
        model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate), loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),metrics=['AUC'])
        train_history=model.fit(Xntrain, YtrainNN,validation_split=0.1,epochs=500,verbose=1, class_weight={0: 1.0, 1:RCC},callbacks=[es])
        tothits,rank,found=tothitscalc(model,'Y',numCF, cf,Xntest)
        NRNN_tot.append(tothits/numCF)
        NRNN_LR.append(rank)
        NRNN_FOUND.append(found)

        tuned_parameters = [{ "ccp_alpha": [0.0, 0.1,1, 10, 100, 1000]}]
        model = GridSearchCV(RandomForestClassifier(class_weight='balanced',max_features=None), tuned_parameters, scoring='roc_auc')
        train_history=model.fit(Xntrain, Ytrain)
        tothits,rank,found=tothitscalc(model,'N',numCF, cf,Xntest)
        RF_tot.append(tothits/numCF)
        RF_LR.append(rank)
        RF_FOUND.append(found)

        model = GridSearchCV(tree.DecisionTreeClassifier(class_weight='balanced'), tuned_parameters, scoring='roc_auc')
        train_history=model.fit(Xntrain, Ytrain)
        tothits,rank,found=tothitscalc(model,'N',numCF, cf,Xntest)
        DT_tot.append(tothits/numCF)
        DT_LR.append(rank)
        DT_FOUND.append(found)
            
    resultsarray.append([NF,NS,RCC,numiter,sum(LASSO_tot)/len(LASSO_tot),(np.std(LASSO_tot))/math.sqrt(len(LASSO_tot)),sum(NN_tot)/len(NN_tot),(np.std(NN_tot))/math.sqrt(len(NN_tot)),sum(DT_tot)/len(DT_tot),(np.std(DT_tot))/math.sqrt(len(DT_tot)),sum(RF_tot)/len(RF_tot),(np.std(RF_tot))/math.sqrt(len(RF_tot)),sum(LASSOSHAP_tot)/len(LASSOSHAP_tot),(np.std(LASSOSHAP_tot))/math.sqrt(len(LASSOSHAP_tot)),sum(NRLASSO_tot)/len(NRLASSO_tot),(np.std(NRLASSO_tot))/math.sqrt(len(NRLASSO_tot)),sum(NRNN_tot)/len(NRNN_tot),(np.std(NRNN_tot))/math.sqrt(len(NRNN_tot)),sum(NRLASSOSHAP_tot)/len(NRLASSOSHAP_tot),(np.std(NRLASSOSHAP_tot))/math.sqrt(len(NRLASSOSHAP_tot))] ) 
    resultsarray2.append([NF,NS,RCC,numiter,sum(LASSO_LR)/len(LASSO_LR),(np.std(LASSO_LR))/math.sqrt(len(LASSO_LR)),sum(NN_LR)/len(NN_LR),(np.std(NN_LR))/math.sqrt(len(NN_LR)),sum(DT_LR)/len(DT_LR),(np.std(DT_LR))/math.sqrt(len(DT_LR)),sum(RF_LR)/len(RF_LR),(np.std(RF_LR))/math.sqrt(len(RF_LR)),sum(LASSOSHAP_LR)/len(LASSOSHAP_LR),(np.std(LASSOSHAP_LR))/math.sqrt(len(LASSOSHAP_LR)),sum(NRLASSO_LR)/len(NRLASSO_LR),(np.std(NRLASSO_LR))/math.sqrt(len(NRLASSO_LR)),sum(NRNN_LR)/len(NRNN_LR),(np.std(NRNN_LR))/math.sqrt(len(NRNN_LR)),sum(NRLASSOSHAP_LR)/len(NRLASSOSHAP_LR),(np.std(NRLASSOSHAP_LR))/math.sqrt(len(NRLASSOSHAP_LR))] )
    resultsarray3.append([NF,NS,RCC,numiter,sum(LASSO_FOUND)/len(LASSO_FOUND),(np.std(LASSO_FOUND))/math.sqrt(len(LASSO_FOUND)),sum(NN_FOUND)/len(NN_FOUND),(np.std(NN_FOUND))/math.sqrt(len(NN_FOUND)),sum(DT_FOUND)/len(DT_FOUND),(np.std(DT_FOUND))/math.sqrt(len(DT_FOUND)),sum(RF_FOUND)/len(RF_FOUND),(np.std(RF_FOUND))/math.sqrt(len(RF_FOUND)),sum(LASSOSHAP_FOUND)/len(LASSOSHAP_FOUND),(np.std(LASSOSHAP_FOUND))/math.sqrt(len(LASSOSHAP_FOUND)),sum(NRLASSO_FOUND)/len(NRLASSO_FOUND),(np.std(NRLASSO_FOUND))/math.sqrt(len(NRLASSO_FOUND)),sum(NRNN_FOUND)/len(NRNN_FOUND),(np.std(NRNN_FOUND))/math.sqrt(len(NRNN_FOUND)),sum(NRLASSOSHAP_FOUND)/len(NRLASSOSHAP_FOUND),(np.std(NRLASSOSHAP_FOUND))/math.sqrt(len(NRLASSOSHAP_FOUND))] )

resultsarraytitle=[['NF','NS','RCC','number iterations','LR','LR_err','NN','NN_err','DT','DT_err','RF','RF err','LR shap','LR shap err','LR_noreg','LR_noreg_err','NN_noreg','NN_noreg_err','LR_noreg SHAP','LR_noreg SHAP err']]
forcsv=resultsarraytitle+resultsarray
with open('results/cnv Hits@k by dataset size scenario '+str(GeneticScenario)+str(numCF)+'.csv','w', newline='') as result_file:
    wr = csv.writer(result_file, dialect='excel')
    wr.writerows(forcsv) 

forcsv=resultsarraytitle+resultsarray2
with open('results/cnv last rank by dataset size scenario '+str(GeneticScenario)+str(numCF)+'.csv','w', newline='') as result_file:
    wr = csv.writer(result_file, dialect='excel')
    wr.writerows(forcsv) 

forcsv=resultsarraytitle+resultsarray3
with open('results/cnv Found by dataset size scenario '+str(GeneticScenario)+str(numCF)+'.csv','w', newline='') as result_file:
    wr = csv.writer(result_file, dialect='excel')
    wr.writerows(forcsv) 
    
plt.clf
resultsarray=np.array(resultsarray)
ax = plt.gca()
ax.set_ylim([0, 1.05])

plt.errorbar(NSlistlog,resultsarray[0:len(NSlist),4],yerr=resultsarray[0:len(NSlist),5],color='red', linestyle='dashed', markerfacecolor='red')
plt.errorbar(NSlistlog,resultsarray[0:len(NSlist),12],yerr=resultsarray[0:len(NSlist),13],color='pink', linestyle='dashed',markerfacecolor='pink')
plt.errorbar(NSlistlog,resultsarray[0:len(NSlist),6],yerr=resultsarray[0:len(NSlist),7],color='green', linestyle='dashed', markerfacecolor='green')
plt.errorbar(NSlistlog,resultsarray[0:len(NSlist),8],yerr=resultsarray[0:len(NSlist),9],color='blue',markerfacecolor='blue')
plt.errorbar(NSlistlog,resultsarray[0:len(NSlist),10],yerr=resultsarray[0:len(NSlist),11],color='orange',markerfacecolor='orange')
plt.errorbar(NSlistlog,resultsarray[0:len(NSlist),18],yerr=resultsarray[0:len(NSlist),19],color='pink',markerfacecolor='pink')
plt.errorbar(NSlistlog,resultsarray[0:len(NSlist),14],yerr=resultsarray[0:len(NSlist),15],color='red', markerfacecolor='red')
plt.errorbar(NSlistlog,resultsarray[0:len(NSlist),16],yerr=resultsarray[0:len(NSlist),17],color='green', markerfacecolor='green')

ax.xaxis.set_major_formatter(ScalarFormatter())
plt.xticks(ticks=NSlistlog, labels=['10000','1000','500','100','50'])
plt.legend(['LR+L1 coefs', 'LR+L1 SHAP','NN+L1 SHAP','DT SHAP','RF SHAP','LR coefs','LR SHAP','NN SHAP'])
plt.xlabel('Number of samples') 
plt.ylabel('Fraction of causative features') 
plt.title('Hits@'+str(numCF)+' by dataset size (scenario '+str(GeneticScenario)+str(numCF)+')') 
plt.savefig('results/cnv Hits@'+str(numCF)+' by dataset size (scenario '+str(GeneticScenario)+str(numCF)+').jpg')
plt.clf()
plt.cla()
plt.close()

plt.clf
resultsarray=np.array(resultsarray2)
ax = plt.gca()
ax.set_ylim([0, NF*1.01])

plt.errorbar(NSlistlog,resultsarray[0:len(NSlist),4],yerr=resultsarray[0:len(NSlist),5],color='red', linestyle='dashed', markerfacecolor='red')
plt.errorbar(NSlistlog,resultsarray[0:len(NSlist),12],yerr=resultsarray[0:len(NSlist),13],color='pink', linestyle='dashed',markerfacecolor='pink')
plt.errorbar(NSlistlog,resultsarray[0:len(NSlist),6],yerr=resultsarray[0:len(NSlist),7],color='green', linestyle='dashed', markerfacecolor='green')
plt.errorbar(NSlistlog,resultsarray[0:len(NSlist),8],yerr=resultsarray[0:len(NSlist),9],color='blue',markerfacecolor='blue')
plt.errorbar(NSlistlog,resultsarray[0:len(NSlist),10],yerr=resultsarray[0:len(NSlist),11],color='orange',markerfacecolor='orange')
plt.errorbar(NSlistlog,resultsarray[0:len(NSlist),18],yerr=resultsarray[0:len(NSlist),19],color='pink',markerfacecolor='pink')
plt.errorbar(NSlistlog,resultsarray[0:len(NSlist),14],yerr=resultsarray[0:len(NSlist),15],color='red', markerfacecolor='red')
plt.errorbar(NSlistlog,resultsarray[0:len(NSlist),16],yerr=resultsarray[0:len(NSlist),17],color='green', markerfacecolor='green')

ax.xaxis.set_major_formatter(ScalarFormatter())
plt.xticks(ticks=NSlistlog, labels=['10000','1000','500','100','50'])
plt.legend(['LR+L1 coefs', 'LR+L1 SHAP','NN+L1 SHAP','DT SHAP','RF SHAP','LR coefs','LR SHAP','NN SHAP'])
plt.xlabel('Number of samples') 
plt.ylabel('Last rank') 
plt.title('Last rank by dataset size (scenario '+str(GeneticScenario)+str(numCF)+')') 
plt.savefig('results/cnv Last rank by dataset size (scenario '+str(GeneticScenario)+str(numCF)+').jpg')
plt.clf()
plt.cla()
plt.close()

plt.clf
resultsarray=np.array(resultsarray3)
ax = plt.gca()
ax.set_ylim([0, numCF+1])

plt.errorbar(NSlistlog,resultsarray[0:len(NSlist),4],yerr=resultsarray[0:len(NSlist),5],color='red', linestyle='dashed', markerfacecolor='red')
plt.errorbar(NSlistlog,resultsarray[0:len(NSlist),12],yerr=resultsarray[0:len(NSlist),13],color='pink', linestyle='dashed',markerfacecolor='pink')
plt.errorbar(NSlistlog,resultsarray[0:len(NSlist),6],yerr=resultsarray[0:len(NSlist),7],color='green', linestyle='dashed', markerfacecolor='green')
plt.errorbar(NSlistlog,resultsarray[0:len(NSlist),8],yerr=resultsarray[0:len(NSlist),9],color='blue',markerfacecolor='blue')
plt.errorbar(NSlistlog,resultsarray[0:len(NSlist),10],yerr=resultsarray[0:len(NSlist),11],color='orange',markerfacecolor='orange')
plt.errorbar(NSlistlog,resultsarray[0:len(NSlist),18],yerr=resultsarray[0:len(NSlist),19],color='pink',markerfacecolor='pink')
plt.errorbar(NSlistlog,resultsarray[0:len(NSlist),14],yerr=resultsarray[0:len(NSlist),15],color='red', markerfacecolor='red')
plt.errorbar(NSlistlog,resultsarray[0:len(NSlist),16],yerr=resultsarray[0:len(NSlist),17],color='green', markerfacecolor='green')

ax.xaxis.set_major_formatter(ScalarFormatter())
plt.xticks(ticks=NSlistlog, labels=['10000','1000','500','100','50'])
plt.legend(['LR+L1 coefs', 'LR+L1 SHAP','NN+L1 SHAP','DT SHAP','RF SHAP','LR coefs','LR SHAP','NN SHAP'])
plt.xlabel('Number of samples') 
plt.ylabel('Number CFs found') 
plt.title('Number CFs found by dataset size (scenario '+str(GeneticScenario)+str(numCF)+')') 
plt.savefig('results/cnv Number CFs found by dataset size (scenario '+str(GeneticScenario)+str(numCF)+').jpg')
plt.clf()
plt.cla()
plt.close()