import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.model_selection import train_test_split


def trainModel():
    data = pd.read_csv('predictionApi/liver_disease.csv')
    cols = data.columns
    X = data.drop(['Dataset'], axis=1)
    y = data['Dataset']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train, y_train)
    with open('predictionApi/model.pkl', 'wb') as fh:
        pickle.dump(random_forest, fh)

def getcols():
    data = pd.read_csv('predictionApi/liver_disease.csv')
    cols = data.columns
    return list(cols[:-1])

def prediction(request):
    test = pd.DataFrame([[0] * (len(getcols()))], columns=getcols())

    syms = request['symptoms']
    print(syms)
    for key in syms:
        print(key)
        test[key] = float(syms[key])

    try:
        with open('predictionApi/model.pkl', 'rb') as file:
            model = pickle.load(file)
    except:
        trainModel()
        with open('predictionApi/model.pkl', 'rb') as file:
            model = pickle.load(file)

    if(model.predict(test) == 1):
        return "have disease"
    else:
        return "not have disease"


