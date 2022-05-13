# -*- coding: utf-8 -*-
"""
Created on Fri May  6 17:55:18 2022

@author: berkay
"""

#Classifiers 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier



from sklearn.model_selection import train_test_split,cross_val_score,cross_validate
from sklearn.metrics import confusion_matrix,make_scorer
from sklearn.metrics import recall_score,precision_score
#from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
from pandas import read_csv
import seaborn as sn


url = "diabetes-dataset.csv"
dataset = read_csv(url)

array = dataset.values
x = array[:,0:8]
y = array[:,8]



x_train, x_test, y_train, y_test =train_test_split(x,y,train_size=0.7,test_size=0.3,random_state=0,stratify=y)



clfDTC =  DecisionTreeClassifier(random_state=(0))
clfRFC = RandomForestClassifier(max_depth=14, random_state=0) #we obtained best result with max_depth = 14 or 15
clfMLP = MLPClassifier(solver='adam',random_state=1, max_iter=100000) # Iteration sayısı artıkça accuracy de artıyor

#--------------------------------------------------------------------------------------

print(" ")
print("----> Decision Tree Classifier:")
print(" ")

#Decision Tree Classifier
clfDTC.fit(x_train,y_train)
test_sonuc1 = clfDTC.predict(x_test)

print("10 Fold Cross Validation Scores of Decision Tree Classifier:") 
scores1 = cross_val_score(clfRFC,x,y,cv=10)
print(scores1)
print(" ")
print("Accuracy -->  %0.2f" % (scores1.mean()))
print("Recall Score --> " + str(recall_score(test_sonuc1,y_test)))
print("Precision Score --> " + str(precision_score(test_sonuc1,y_test)))
print(" ")
#print('The accuracy_score of Decision Tree Classifier : ' + str(accuracy_score(test_sonuc1,y_test)))


print("Confusion Matrix")
#print(classification_report(y_test,test_sonuc1))
cm = confusion_matrix(y_test,test_sonuc1)
print(cm)
plt.matshow(cm)
sn.heatmap(cm, annot=True,cmap="OrRd")
plt.title('Confusion matrix for Decision Tree')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

#Data from user to predict
print("-----------------------------------------------------------------------------")
Result1= clfDTC.predict([[2,138,62,35,0,33.6,0.127,47]]) 
print("The prediction using Decision Tree Classifier : ")
print (Result1)
print("-----------------------------------------------------------------------------")


print(" ")
print("---> Random Forest Classifier:")
print(" ")

#Random Forest Classifier
clfRFC.fit(x_train,y_train)
test_sonuc2 = clfRFC.predict(x_test)

print("10 Fold Cross Validation Scores of Random Forest Classifier:") 
scores2 = cross_val_score(clfRFC,x,y,cv=10)
print(scores2)
print(" ")
print("Accuracy -->  %0.2f" % (scores2.mean()))
print("Recall Score --> " + str(recall_score(test_sonuc2,y_test)))
print("Precision Score --> " + str(precision_score(test_sonuc2,y_test)))
print(" ")
#print('The accuracy_score of Decision Tree Classifier : ' + str(accuracy_score(test_sonuc2,y_test)))


print("Confusion Matrix")
#print(classification_report(y_test,test_sonuc2))
cm = confusion_matrix(y_test,test_sonuc2)
print(cm)
plt.matshow(cm)
sn.heatmap(cm, annot=True,cmap="OrRd")
plt.title('Confusion matrix for Random Forest')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

#Data from user to predict
print("-----------------------------------------------------------------------------")
Result2= clfRFC.predict([[2,138,62,35,0,33.6,0.127,47]]) 
print("The prediction using Random Forest Classifier : ")
print (Result2)
print("-----------------------------------------------------------------------------")



# MLP Classifier  -> Deep Learning kapsamındadır
# Veri sayısı az kaldığı için geleneksel makine öğrenmesinin sonuçlarından daha kötü değerler aldığımızı düşünüyoruz
# Ayrıca diğer classifier'lara oranla daha yavaş çalıştı. 
print(" ")
print("----> MLP Classifier:")
print(" ")

clfMLP.fit(x_train,y_train)
test_sonuc3 = clfMLP.predict(x_test)

print("10 Fold Cross Validation Scores of MLP Classifier:") 
scores3  = cross_val_score(clfMLP,x,y,cv=10)
print(scores3)
print(" ")
print("Accuracy -->  %0.2f" % (scores3.mean()))
print("Recall Score --> " + str(recall_score(test_sonuc3,y_test)))
print("Precision Score --> " + str(precision_score(test_sonuc3,y_test)))
print("")
#print('The accuracy_score of MLP Classifier : ' + str(accuracy_score(test_sonuc3,y_test)))

print("Confusion Matrix")
#(classification_report(y_test,test_sonuc3))
cm = confusion_matrix(y_test,test_sonuc3)
print(cm)
plt.matshow(cm)
sn.heatmap(cm, annot=True,cmap="OrRd")
plt.title('Confusion matrix for MLP')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

#Data from user to predict
print("-----------------------------------------------------------------------------")
Result3= clfMLP.predict([[2,138,62,35,0,33.6,0.127,47]]) 
print("The prediction using MLP Classifier : ")
print (Result3)
print("-----------------------------------------------------------------------------")

scoring = {'prec_macro': 'precision_macro',
           'rec_macro': make_scorer(recall_score, average='macro')}

scores = cross_validate(clfMLP, x, y, scoring=scoring,
                        cv=10, return_train_score=True)
sorted(scores.keys())
print(scores['train_rec_macro'])
print("Accuracy -->  %0.2f" % (scores['train_rec_macro'].mean()))



