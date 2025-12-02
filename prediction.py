import joblib
import numpy as np

clf = joblib.load("rf_model.sav")

def get_classes():

    if hasattr(clf, "classes_"):
        return list(clf.classes_)
    return []

def predict_with_proba(input_data):

    data = np.array(input_data).reshape(1, -1)
    pred = clf.predict(data)[0]
    probs = None
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(data)[0]  # array shape (n_classes,)
    return pred, probs
