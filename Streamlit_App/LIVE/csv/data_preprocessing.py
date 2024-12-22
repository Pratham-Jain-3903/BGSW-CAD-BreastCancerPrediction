import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(file_path, target_variable):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Split the dataset into features and target
    X = data.drop(columns=[target_variable])
    y = data[target_variable]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Save the training and testing sets
    X_train.to_csv('train_data.csv', index=False)
    X_test.to_csv('test_data.csv', index=False)
    y_train.to_csv('train_labels.csv', index=False)
    y_test.to_csv('test_labels.csv', index=False)

    return X_train, X_test, y_train, y_test
