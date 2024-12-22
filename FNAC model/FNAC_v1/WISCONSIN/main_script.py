import os

# Get target variable and CSV file from the user
target_variable = input("Enter the name of the target variable: ")
csv_filename = input("Enter the name of the CSV file (with extension e.g., .csv): ")

# Run data preprocessing
os.system(f'python data_preprocessing.py --target {target_variable} --csv_filename {csv_filename}')

# Run feature selection
os.system(f'python feature_selection.py --target {target_variable}')

# Run model training and evaluation
os.system(f'python model_training_evaluation.py --target {target_variable}')

print("The process has been completed. Please provide the values for the 9 selected features for prediction.")
