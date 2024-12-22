import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
import joblib

def train_and_evaluate_model(selected_features):
    X_train = pd.read_csv('train_data.csv')
    X_test = pd.read_csv('test_data.csv')
    y_train = pd.read_csv('train_labels.csv')
    y_test = pd.read_csv('test_labels.csv')

    X_train = X_train[selected_features]
    X_test = X_test[selected_features]

    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    clf1 = RandomForestClassifier(random_state=42)
    clf2 = GradientBoostingClassifier(random_state=42)
    clf3 = LogisticRegression(random_state=42)
    ensemble_model = VotingClassifier(estimators=[('rf', clf1), ('gb', clf2), ('lr', clf3)], voting='soft')

    skf = StratifiedKFold(n_splits=5)
    cv_scores = cross_val_score(ensemble_model, X_train_balanced, y_train_balanced.values.ravel(), cv=skf, scoring='accuracy')

    ensemble_model.fit(X_train_balanced, y_train_balanced.values.ravel())
    y_pred = ensemble_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    joblib.dump(ensemble_model, 'models/trained_ensemble_model.pkl')

    return accuracy, cv_scores
