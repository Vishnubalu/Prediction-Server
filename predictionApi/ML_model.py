
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, KFold
import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, KFold


def trainModel():
    df = pd.read_csv("data/liver_disease.csv")
    df.drop(['Direct_Bilirubin', 'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin'], axis=1, inplace=True)
    df.Albumin_and_Globulin_Ratio.fillna(df.Albumin_and_Globulin_Ratio.mean(), inplace=True)
    skewed_cols = ['Albumin_and_Globulin_Ratio', 'Total_Bilirubin', 'Alkaline_Phosphotase', 'Alamine_Aminotransferase']
    for c in skewed_cols:
        df[c] = df[c].apply('log1p')

    minority = df[df.Dataset == 2]
    majority = df[df.Dataset == 1]

    minority_upsample = resample(minority, replace=True, n_samples=majority.shape[0])

    df = pd.concat([minority_upsample, majority], axis=0)
    rs = RobustScaler()
    for c in df[['Age', 'Gender', 'Total_Bilirubin', 'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
                 'Albumin_and_Globulin_Ratio']].columns:
        df[c] = rs.fit_transform(df[c].values.reshape(-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(df.drop('Dataset', axis=1), df['Dataset'], test_size=0.25,
                                                        random_state=123)
    params = {
        'n_estimators': [100, 200, 500],
        'criterion': ['gini', 'entropy'],
        'min_samples_split': [2, 4, 5],
        'min_samples_leaf': [1, 2, 4, 5],
        'max_leaf_nodes': [4, 10, 20, 50, None]
    }

    gs1 = GridSearchCV(RandomForestClassifier(n_jobs=-1), params, n_jobs=-1, cv=KFold(n_splits=3), scoring='roc_auc')
    gs1.fit(X_train, y_train)
    with open('data/model.pkl', 'wb') as fh:
        pickle.dump(gs1, fh)

def getcols():
    data = pd.read_csv('data/liver_disease.csv')
    cols = data.columns
    return list(cols[:-1])

def prediction(request):
    test = pd.DataFrame([[0] * (len(getcols()))], columns=getcols())
    syms = request['symptoms']

    for key in syms:
        if(key != "Gender"):
            test[key] = float(syms[key])
        else:
            if syms[key] == '1':
                test[key] = "Male"
            else:
                test[key] = "Female"

    test = transform_data(test)
    try:
        with open('data/model.pkl', 'rb') as file:
            model = pickle.load(file)
    except:
        trainModel()
        with open('data/model.pkl', 'rb') as file:
            model = pickle.load(file)

    if(model.predict(test) == 1):
        return "have disease"
    else:
        return "not have disease"


def transform_data(request):
    print("transform data")
    request.drop(['direct_bilirubin', 'aspartate_aminotransferase', 'total_protiens', 'albumin'], axis=1, inplace=True)
    new_gender = {"Male": 1, "Female": 0}
    request["gender"] = request.gender.map(new_gender)
    skewed_cols = ['albumin_and_globulin_ratio', 'total_bilirubin', 'alkaline_phosphotase', 'alamine_aminotransferase']

    for c in skewed_cols:
        request[c] = request[c].apply('log1p')

    rs = RobustScaler()
    for c in request[['age', 'gender', 'total_bilirubin', 'alkaline_phosphotase', 'alamine_aminotransferase',
                      'albumin_and_globulin_ratio']].columns:
        request[c] = rs.fit_transform(request[c].values.reshape(-1, 1))
    return request


def predictFromCSV(dataframe):
    dataframe.columns = map(str.lower, dataframe.columns)
    cols = getcols()
    cols = list(map(str.lower, cols))
    print("cols ", cols)
    for col in cols:
        if(col not in list(dataframe.columns)):
            return ['incorrect']
    dataframe = dataframe[cols]
    print("columns ", dataframe.columns)
    df = transform_data(dataframe)
    try:
        with open('data/model.pkl', 'rb') as file:
            model = pickle.load(file)
    except:
        trainModel()
        with open('data/model.pkl', 'rb') as file:
            model = pickle.load(file)

    ans = model.predict(df)
    return ans


