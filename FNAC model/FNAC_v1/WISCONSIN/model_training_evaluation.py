# import argparse
# from sklearn.ensemble import VotingClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import StratifiedKFold
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
# import pandas as pd
# import matplotlib.pyplot as plt
# from reportlab.lib.pagesizes import letter
# from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
# from reportlab.lib.styles import getSampleStyleSheet
# from reportlab.lib import colors

# # Function to extract metrics from the classification report
# def extract_classification_metrics(classification_rep):
#     metrics = []
    
#     lines = classification_rep.split('\n')[2:-3]

#     for line in lines:
#         row = line.split()
#         if row and len(row) == 5:
#             class_label = row[0]
#             precision = float(row[1])
#             recall = float(row[2])
#             f1_score = float(row[3])
#             support = int(row[4])
            
#             metrics.append({
#                 "Class": class_label,
#                 "Precision": precision,
#                 "Recall": recall,
#                 "F1-Score": f1_score,
#                 "Support": support
#             })

#     return metrics

# # Function to create ROC curve
# def create_roc_curve(X, y, model):
#     y_numeric = (y == 'M').astype(int)

#     model.fit(X, y_numeric)

#     y_probs = model.predict_proba(X)[:, 1]

#     fpr, tpr, _ = roc_curve(y_numeric, y_probs)
#     roc_auc = auc(fpr, tpr)

#     plt.figure(figsize=(8, 6))
#     plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
#     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic (ROC) Curve')
#     plt.legend(loc="lower right")
    
#     roc_curve_filename = 'roc_curve.png'
#     plt.savefig(roc_curve_filename)
#     plt.close()

#     return roc_curve_filename

# # Function to create PDF report
# def create_pdf_report(pdf_filename, target_variable, model_name, accuracy, classification_rep, conf_matrix_plot_filename, X, y, model):
#     doc = SimpleDocTemplate(pdf_filename, pagesize=letter)

#     content = []

#     title_style = getSampleStyleSheet()["Title"]
#     content.append(Paragraph(f"<u>Model Evaluation Report for {target_variable}</u>", title_style))
#     content.append(Spacer(1, 12))

#     heading_style = getSampleStyleSheet()["Heading1"]
#     content.append(Paragraph(f"<b>Chosen Model:</b> {model_name}", heading_style))
#     content.append(Spacer(1, 12))

#     content.append(Paragraph(f"<b>Accuracy on Training Set:</b> {accuracy:.4f}", heading_style))
#     content.append(Spacer(1, 12))

#     content.append(Paragraph("<b>Classification Report:</b>", heading_style))

#     metrics = extract_classification_metrics(classification_rep)
#     table_style = [('GRID', (0, 0), (-1, -1), 1, colors.black),
#                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
#                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
#                    ('ALIGN', (0, 0), (-1, -1), 'CENTER')]
#     table_data = [["Class", "Precision", "Recall", "F1-Score", "Support"]]
#     for metric in metrics:
#         table_data.append([metric["Class"], f"{metric['Precision']:.2f}", f"{metric['Recall']:.2f}", f"{metric['F1-Score']:.2f}", metric["Support"]])

#     content.append(Spacer(1, 6))
#     content.append(Table(table_data, style=table_style))
#     content.append(Spacer(1, 12))

#     content.append(Paragraph("<b>Confusion Matrix Plot:</b>", heading_style))
#     content.append(Image(conf_matrix_plot_filename, width=400, height=300))

#     roc_curve_filename = create_roc_curve(X, y, model)
#     content.append(Paragraph("<b>ROC Curve:</b>", heading_style))
#     content.append(Image(roc_curve_filename, width=400, height=300))

#     doc.build(content)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--target', type=str, required=True, help='Name of the target variable')
#     args = parser.parse_args()

#     # Load the data
#     train_data = pd.read_csv(r'C:\Users\Pratham Jain\SisterDear\AIHC\Atharva\train_data.csv')
#     selected_features_info = pd.read_csv(r'C:\Users\Pratham Jain\SisterDear\AIHC\Atharva\selected_features_info.csv')

#     X_train = train_data[selected_features_info['Feature Name']]
#     y_train = train_data[args.target]

#     # Scale the features
#     scaler = MinMaxScaler()
#     X_train_scaled = scaler.fit_transform(X_train)

#     # Define models for ensemble
#     models = {
#         'Logistic Regression': LogisticRegression(),
#         'Decision Tree Classifier': DecisionTreeClassifier(),
#         'Random Forest Classifier': RandomForestClassifier(),
#         'Support Vector Classifier': SVC(probability=True),
#         'K-Nearest Neighbors': KNeighborsClassifier()
#     }

#     # Define the ensemble model
#     ensemble_model = VotingClassifier(estimators=[(name, model) for name, model in models.items()], voting='soft')

#     # Perform Stratified K-Fold Cross-Validation
#     cv = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)
#     cv_scores = []
#     for train_idx, val_idx in cv.split(X_train_scaled, y_train):
#         X_train_cv, X_val_cv = X_train_scaled[train_idx], X_train_scaled[val_idx]
#         y_train_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]
#         ensemble_model.fit(X_train_cv, y_train_cv)
#         cv_scores.append(ensemble_model.score(X_val_cv, y_val_cv))

#     mean_accuracy = sum(cv_scores) / len(cv_scores)
#     print(f"Cross-validated accuracy score: {mean_accuracy:.4f}")

    # # Prompt user to input the 9 best feature values
    # print("Please provide the values for the 9 selected features for prediction:")
    # feature_values = []
    # for feature in selected_features_info['Feature Name']:
    #     value = float(input(f"Enter value for {feature}: "))
    #     feature_values.append(value)

#     # Directly provide the feature values for prediction
#     feature_values = [14.42, 94.48, 642.5, 2.376, 26.85, 16.33, 30.86, 109.5, 826.4]

    
#     X_input = [feature_values]
#     X_input_scaled = scaler.transform(X_input)
#     y_pred = ensemble_model.predict(X_input)
#     prediction = "Malignant" if y_pred[0] == 'M' else "Benign"
#     print()
#     print(f"The prediction based on the provided features is: {prediction}")
#     print()

#     # Generate the classification report and confusion matrix
#     y_train_numeric = (y_train == 'M').astype(int)
#     y_pred_train = ensemble_model.predict(X_train_scaled)
#     classification_rep = classification_report(y_train_numeric, (ensemble_model.predict(X_train_scaled) == 'M').astype(int))
#     conf_matrix = confusion_matrix(y_train_numeric, (ensemble_model.predict(X_train_scaled) == 'M').astype(int))
#     conf_matrix_plot_filename = 'confusion_matrix.png'
#     ConfusionMatrixDisplay(conf_matrix, display_labels=['Benign', 'Malignant']).plot(cmap='Blues')
#     plt.title('Confusion Matrix')
#     plt.savefig(conf_matrix_plot_filename)
#     plt.close()

#     # Create a PDF report
#     pdf_filename = r'C:\Users\Pratham Jain\SisterDear\AIHC\Atharva\model_evaluation_report.pdf'
#     create_pdf_report(pdf_filename, args.target, 'Voting Classifier Ensemble', mean_accuracy, classification_rep, conf_matrix_plot_filename, X_train, y_train, ensemble_model)

#     print(f"Model evaluation completed. Report saved to {pdf_filename}.")

import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors


# Function to extract metrics from the classification report
def extract_classification_metrics(classification_rep):
    metrics = []
    
    lines = classification_rep.split('\n')[2:-3]

    for line in lines:
        row = line.split()
        if row and len(row) == 5:
            class_label = row[0]
            precision = float(row[1])
            recall = float(row[2])
            f1_score = float(row[3])
            support = float(row[4])  # Changed to float
            
            metrics.append({
                "Class": class_label,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1_score,
                "Support": support
            })

    return metrics

# Function to create ROC curve
def create_roc_curve(X, y, model):
    y_numeric = (y == 'M').astype(int)

    model.fit(X, y_numeric)

    y_probs = model.predict_proba(X)[:, 1]

    fpr, tpr, _ = roc_curve(y_numeric, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    roc_curve_filename = 'roc_curve.png'
    plt.savefig(roc_curve_filename)
    plt.close()

    return roc_curve_filename

def create_pdf_report(pdf_filename, target_variable, model_name, train_accuracy, test_accuracy, train_classification_rep, test_classification_rep, train_conf_matrix_filename, test_conf_matrix_filename, X_train, y_train, X_test, y_test, model):
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
    content = []

    title_style = getSampleStyleSheet()["Title"]
    content.append(Paragraph(f"<u>Model Evaluation Report for {target_variable}</u>", title_style))
    content.append(Spacer(1, 12))

    heading_style = getSampleStyleSheet()["Heading1"]
    content.append(Paragraph(f"<b>Chosen Model:</b> {model_name}", heading_style))
    content.append(Spacer(1, 12))

    content.append(Paragraph(f"<b>Accuracy on Training Set:</b> {train_accuracy:.4f}", heading_style))
    content.append(Paragraph(f"<b>Prediction</b> {prediction}", heading_style))
    content.append(Spacer(1, 12))

    for dataset, classification_rep, conf_matrix_filename in [("Training", train_classification_rep, train_conf_matrix_filename), 
                                                              ("Test", test_classification_rep, test_conf_matrix_filename)]:
        content.append(Paragraph(f"<b>{dataset} Set Classification Report:</b>", heading_style))
        metrics = extract_classification_metrics(classification_rep)
        if metrics:
            table_style = [('GRID', (0, 0), (-1, -1), 1, colors.black),
                           ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                           ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                           ('ALIGN', (0, 0), (-1, -1), 'CENTER')]
            table_data = [["Class", "Precision", "Recall", "F1-Score", "Support"]]
            for metric in metrics:
                table_data.append([metric["Class"], f"{metric['Precision']:.2f}", f"{metric['Recall']:.2f}", f"{metric['F1-Score']:.2f}", f"{metric['Support']:.0f}"])
            content.append(Spacer(1, 6))
            content.append(Table(table_data, style=table_style))
        else:
            content.append(Paragraph("Not enough data to generate classification report.", heading_style))
        content.append(Spacer(1, 12))

        content.append(Paragraph(f"<b>{dataset} Set Confusion Matrix:</b>", heading_style))
        content.append(Image(conf_matrix_filename, width=400, height=300))
        content.append(Spacer(1, 12))

    for dataset, X, y in [("Training", X_train, y_train), ("Test", X_test, y_test)]:
        if len(np.unique(y)) > 1:  # Only create ROC curve if there are multiple classes
            roc_curve_filename = create_roc_curve(X, y, model)
            content.append(Paragraph(f"<b>{dataset} Set ROC Curve:</b>", heading_style))
            content.append(Image(roc_curve_filename, width=400, height=300))
        else:
            content.append(Paragraph(f"<b>{dataset} Set ROC Curve:</b> Not applicable (only one class present)", heading_style))
        content.append(Spacer(1, 12))

    doc.build(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, required=True, help='Name of the target variable')
    args = parser.parse_args()

    # Load the data
    train_data = pd.read_csv(r'C:\Users\Pratham Jain\SisterDear\AIHC\Atharva\train_data.csv')
    selected_features_info = pd.read_csv(r'C:\Users\Pratham Jain\SisterDear\AIHC\Atharva\selected_features_info.csv')

    feature_names = selected_features_info['Feature Name']

    X_train = train_data[feature_names]
    y_train = train_data[args.target]

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Define models for ensemble
    models = {
        'Logistic Regression': LogisticRegression(max_iter=5000),
        'Decision Tree Classifier': DecisionTreeClassifier(),
        'Random Forest Classifier': RandomForestClassifier(),
        'Support Vector Classifier': SVC(probability=True),
        'K-Nearest Neighbors': KNeighborsClassifier()
    }

    # Define the ensemble model
    ensemble_model = VotingClassifier(estimators=[(name, model) for name, model in models.items()], voting='soft')

    # Train the model
    ensemble_model.fit(X_train_scaled, y_train)

    # Make predictions on train set
    y_train_pred = ensemble_model.predict(X_train_scaled)

    # Calculate training accuracy
    train_accuracy = accuracy_score(y_train, y_train_pred)

    print(f"Training Accuracy: {train_accuracy:.4f}")

    # Prompt user to input the 9 best feature values
    print("Please provide the values for the 9 selected features for prediction:")
    feature_values = []
    for feature in selected_features_info['Feature Name']:
        value = float(input(f"Enter value for {feature}: "))
        feature_values.append(value)

    # Reshape X_test to be a 2D array
    X_test = np.array(feature_values).reshape(1, -1)

    # Scale the test features
    X_test_scaled = scaler.transform(X_test)

    # Make prediction
    prediction = ensemble_model.predict(X_test_scaled)
    probabilities = ensemble_model.predict_proba(X_test_scaled)

    print(f"\nPredicted label: {prediction[0]}")
    print(f"Probability of being Benign: {probabilities[0][0]:.4f}")
    print(f"Probability of being Malignant: {probabilities[0][1]:.4f}")

    # Generate classification report for training data
    train_classification_rep = classification_report(y_train, y_train_pred)

    # Generate confusion matrix for training data
    train_conf_matrix = confusion_matrix(y_train, y_train_pred)

    # Plot and save confusion matrix
    train_conf_matrix_filename = 'train_confusion_matrix.png'
    plt.figure(figsize=(10, 8))
    ConfusionMatrixDisplay(train_conf_matrix, display_labels=['Benign', 'Malignant']).plot(cmap='Blues')
    plt.title('Training Set Confusion Matrix')
    plt.savefig(train_conf_matrix_filename)
    plt.close()

    # Create a PDF report
    pdf_filename = r'C:\Users\Pratham Jain\SisterDear\AIHC\Atharva\model_evaluation_report.pdf'
    # create_pdf_report(pdf_filename, args.target, 'Voting Classifier Ensemble', 
    #                   train_accuracy, prediction,  # We don't have test accuracy anymore
    #                   train_classification_rep, None,  # We don't have test classification report
    #                   train_conf_matrix_filename, None,  # We don't have test confusion matrix
    #                   X_train_scaled, y_train, X_test_scaled, None, ensemble_model)

    # print(f"\nModel evaluation report saved to {pdf_filename}.")

# Please provide the values for the 9 selected features for prediction:
# Enter value for radius_mean: 19.16
# Enter value for perimeter_mean: 126.2
# Enter value for area_mean: 1138
# Enter value for perimeter_se: 4.321
# Enter value for area_se: 69.65
# Enter value for radius_worst: 23.72
# Enter value for texture_worst: 35.9
# Enter value for perimeter_worst: 159.8
# Enter value for area_worst: 1724
#Expected Prediction : M