import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2

def select_best_features(file_path, target_variable, k=9):
    X_train = pd.read_csv(file_path)
    y_train = pd.read_csv('train_labels.csv')

    if 'id' in X_train.columns:
        X_train = X_train.drop(columns=['id'])

    # Apply SelectKBest
    selector = SelectKBest(score_func=chi2, k=k)
    selector.fit(X_train, y_train.values.ravel())

    selected_features = X_train.columns[selector.get_support()]
    selected_features_df = pd.DataFrame(selected_features, columns=['selected_features'])
    selected_features_df.to_csv('selected_features_info.csv', index=False)

    return selected_features
