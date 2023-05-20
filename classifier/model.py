from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE


def test_split(df, test_size=0.7, is_imbalanced=False):
    X = df.drop(columns='target')
    y = df.target
    if is_imbalanced:
        # define oversampling strategynEA
        over = SMOTE(sampling_strategy="minority")
        # fit and apply the transform
        X, y = over.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=test_size)
    return X_train, X_test, y_train, y_test


def model_selection(X_train, X_test, y_train, y_test):
    dfs = []

    models = [
        ('LogReg', LogisticRegression()),
        ('RF', RandomForestClassifier()),
        ('KNN', KNeighborsClassifier()),
        ('GNB', GaussianNB()),
        ('XGB', XGBClassifier())
    ]
    results = []
    names = []
    scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc']

    for name, model in models:
        kfold = KFold(n_splits=5, shuffle=True)
        cv_results = cross_validate(model, X_train, y_train,
                                    cv=kfold, scoring=scoring)
        clf = model.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(name)
        print(classification_report(y_test, y_pred))
        results.append(cv_results)
        names.append(name)
        df_model = pd.DataFrame(cv_results)
        df_model['model'] = name
        dfs.append(df_model)

    final = pd.concat(dfs, ignore_index=True)
    return final

