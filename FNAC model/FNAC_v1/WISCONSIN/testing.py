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
            support = int(row[4])
            
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
    content.append(Paragraph(f"<b>Accuracy on Test Set:</b> {test_accuracy:.4f}", heading_style))
    content.append(Spacer(1, 12))

    for dataset, classification_rep, conf_matrix_filename in [("Training", train_classification_rep, train_conf_matrix_filename), 
                                                              ("Test", test_classification_rep, test_conf_matrix_filename)]:
        content.append(Paragraph(f"<b>{dataset} Set Classification Report:</b>", heading_style))
        metrics = extract_classification_metrics(classification_rep)
        table_style = [('GRID', (0, 0), (-1, -1), 1, colors.black),
                       ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                       ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                       ('ALIGN', (0, 0), (-1, -1), 'CENTER')]
        table_data = [["Class", "Precision", "Recall", "F1-Score", "Support"]]
        for metric in metrics:
            table_data.append([metric["Class"], f"{metric['Precision']:.2f}", f"{metric['Recall']:.2f}", f"{metric['F1-Score']:.2f}", metric["Support"]])
        content.append(Spacer(1, 6))
        content.append(Table(table_data, style=table_style))
        content.append(Spacer(1, 12))

        content.append(Paragraph(f"<b>{dataset} Set Confusion Matrix:</b>", heading_style))
        content.append(Image(conf_matrix_filename, width=400, height=300))
        content.append(Spacer(1, 12))

    for dataset, X, y in [("Training", X_train, y_train), ("Test", X_test, y_test)]:
        roc_curve_filename = create_roc_curve(X, y, model)
        content.append(Paragraph(f"<b>{dataset} Set ROC Curve:</b>", heading_style))
        content.append(Image(roc_curve_filename, width=400, height=300))
        content.append(Spacer(1, 12))

    doc.build(content)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, required=True, help='Name of the target variable')
    args = parser.parse_args()

    # Load the data
    train_data = pd.read_csv(r'C:\Users\Pratham Jain\SisterDear\AIHC\Atharva\train_data.csv')
    test_data = pd.read_csv(r'C:\Users\Pratham Jain\SisterDear\AIHC\Atharva\test_data.csv')
    selected_features_info = pd.read_csv(r'C:\Users\Pratham Jain\SisterDear\AIHC\Atharva\selected_features_info.csv')

    feature_names = selected_features_info['Feature Name']
    X_train = train_data[feature_names]
    y_train = train_data[args.target]
    X_test = test_data[feature_names]
    y_test = test_data[args.target]

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

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

    # Make predictions on train and test sets
    y_train_pred = ensemble_model.predict(X_train_scaled)
    y_test_pred = ensemble_model.predict(X_test_scaled)

    # Calculate accuracies
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Testing Accuracy: {test_accuracy:.4f}")

    # Generate classification reports
    train_classification_rep = classification_report(y_train, y_train_pred)
    test_classification_rep = classification_report(y_test, y_test_pred)

    # Generate confusion matrices
    train_conf_matrix = confusion_matrix(y_train, y_train_pred)
    test_conf_matrix = confusion_matrix(y_test, y_test_pred)

    # Plot and save confusion matrices
    train_conf_matrix_filename = 'train_confusion_matrix.png'
    test_conf_matrix_filename = 'test_confusion_matrix.png'

    plt.figure(figsize=(10, 8))
    ConfusionMatrixDisplay(train_conf_matrix, display_labels=['Benign', 'Malignant']).plot(cmap='Blues')
    plt.title('Training Set Confusion Matrix')
    plt.savefig(train_conf_matrix_filename)
    plt.close()

    plt.figure(figsize=(10, 8))
    ConfusionMatrixDisplay(test_conf_matrix, display_labels=['Benign', 'Malignant']).plot(cmap='Blues')
    plt.title('Test Set Confusion Matrix')
    plt.savefig(test_conf_matrix_filename)
    plt.close()

    # Create a PDF report
    pdf_filename = r'C:\Users\Pratham Jain\SisterDear\AIHC\Atharva\model_evaluation_report.pdf'
    create_pdf_report(pdf_filename, args.target, 'Voting Classifier Ensemble', 
                      train_accuracy, test_accuracy, 
                      train_classification_rep, test_classification_rep,
                      train_conf_matrix_filename, test_conf_matrix_filename,
                      X_train_scaled, y_train, X_test_scaled, y_test, ensemble_model)

    print(f"Model evaluation completed. Report saved to {pdf_filename}.")
