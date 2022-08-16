import pandas as pd
import unicodedata
import re
import demoji
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.svm import SVC



def removerAcentosECaracteresEspeciais(palavra):

    nfkd = unicodedata.normalize('NFKD', palavra)
    palavraSemAcento = u"".join([c for c in nfkd if not unicodedata.combining(c)])
    
    return re.sub('[^a-zA-Z0-9 \\\]', '', palavraSemAcento)

def removeEmoji(text):
    dem = demoji.findall(text)
    for item in dem.keys():
        text = text.replace(item,'')
    return text

def confusion_matrix_scorer(acc,cm):
      return {
          'Acurácia ': acc,
          'Falso verdadeiro': cm[0, 0], 
          'Falso positivo': cm[0, 1],
          'Falso negativo': cm[1, 0], 
          'Verdadeiro positivo ': cm[1, 1]
      }


def main():
    data = pd.read_csv('tweets.csv',encoding='utf-8',delimiter=';')
    
    data['RateNum'] = data.Rate.map({'Positivo' : 1,'Negativo': -1,'Neutro':0}) 
    x = data.Tweet
    y = data.RateNum
    tweetsTratados = []

    for words in x:
        word = removeEmoji(words)
        word = removerAcentosECaracteresEspeciais(words)
        tweetsTratados.append((word))

    vectorizer = TfidfVectorizer(encoding='utf-8')
    X = vectorizer.fit_transform(tweetsTratados)



    print("Cross - Validation")
    model = MultinomialNB()
    kf = KFold(n_splits=5,shuffle=True)
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        print(confusion_matrix_scorer(metrics.accuracy_score(y_test, predicted),metrics.confusion_matrix(y_test, predicted)))  

    print("Cross - Validation")
    model = MultinomialNB()
    kf = StratifiedKFold(n_splits=5,shuffle=False)
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        print(confusion_matrix_scorer(metrics.accuracy_score(y_test, predicted),metrics.confusion_matrix(y_test, predicted)))  

    model = MultinomialNB().fit(X,y)
    scores = cross_val_score(model,X,y,cv=5)
    print(scores)


    # print('Hold-Out')
    # xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.4)
    # clf = MultinomialNB().fit(xtrain,ytrain)
    # predicted = clf.predict(xtest)
    # print(confusion_matrix_scorer(metrics.accuracy_score(ytest,predicted),metrics.confusion_matrix(ytest,predicted)))

    # print("SVM")
    # xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.3)
    # modelSVC = SVC()
    # modelSVC.fit(xtrain,ytrain)
    # predicted = clf.predict(xtest)
    # print(confusion_matrix_scorer(metrics.accuracy_score(ytest,predicted),metrics.confusion_matrix(ytest,predicted)))  

    # print("Regressão Logística")
    # xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.3)
    # modelRL = LogisticRegression(random_state=16)
    # modelRL.fit(xtrain,ytrain)
    # predicted = clf.predict(xtest)
    # print(confusion_matrix_scorer(metrics.accuracy_score(ytest,predicted),metrics.confusion_matrix(ytest,predicted)))


    # print("Cross - Validation - SVM")
    # model = SVC()
    # kf = StratifiedKFold(n_splits=5)
    # for train_index, test_index in kf.split(X, y):
    #     X_train, X_test = X[train_index], X[test_index]
    #     y_train, y_test = y[train_index], y[test_index]
    #     model.fit(X_train, y_train)
    #     predicted = model.predict(X_test)
    #     print(confusion_matrix_scorer(metrics.accuracy_score(y_test, predicted),metrics.confusion_matrix(y_test, predicted)))  

    # print("Cross - Validation - Regressão Logística")
    # model = LogisticRegression(random_state=16)
    # kf = StratifiedKFold(n_splits=5)
    # for train_index, test_index in kf.split(X, y):
    #     X_train, X_test = X[train_index], X[test_index]
    #     y_train, y_test = y[train_index], y[test_index]
    #     model.fit(X_train, y_train)
    #     predicted = model.predict(X_test)
    #     print(confusion_matrix_scorer(metrics.accuracy_score(y_test, predicted),metrics.confusion_matrix(y_test, predicted)))  

main()