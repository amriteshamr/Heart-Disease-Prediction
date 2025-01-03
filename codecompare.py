import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
heart_data = pd.read_csv('heart.csv')

# Select features based on correlation with target
selected_features = ['cp', 'thalach', 'slope', 'restecg', 'sex', 'thal', 'ca', 'exang', 'oldpeak']
X_selected = heart_data[selected_features]
y = heart_data['target']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)

# Logistic Regression Model
logreg = LogisticRegression(max_iter=500, random_state=42)
logreg.fit(X_train, y_train)
logreg_pred = logreg.predict(X_test)
logreg_accuracy = accuracy_score(y_test, logreg_pred)

# SVM Model
svm_model = SVC(kernel='rbf', C=1, random_state=42)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_pred)

# Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=10, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)

# Collecting performance metrics
models = ['Logistic Regression', 'SVM', 'Random Forest']
accuracies = [logreg_accuracy, svm_accuracy, rf_accuracy]

# Plotting accuracy comparison
plt.figure(figsize=(8, 5))
plt.bar(models, accuracies, color=['blue', 'green', 'red'])
plt.title('Model Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.show()

# Function to plot confusion matrix
def plot_confusion_matrix(model, X_test, y_test, model_name):
    cm = confusion_matrix(y_test, model.predict(X_test))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Plot Confusion Matrices for each model
plot_confusion_matrix(logreg, X_test, y_test, 'Logistic Regression')
plot_confusion_matrix(svm_model, X_test, y_test, 'SVM')
plot_confusion_matrix(rf_model, X_test, y_test, 'Random Forest')

# Classification Report Metrics Comparison
logreg_report = classification_report(y_test, logreg_pred, output_dict=True)
svm_report = classification_report(y_test, svm_pred, output_dict=True)
rf_report = classification_report(y_test, rf_pred, output_dict=True)

# Extract Precision, Recall, and F1-Score for each class
metrics = ['precision', 'recall', 'f1-score']
logreg_metrics = [logreg_report['1'][metric] for metric in metrics]
svm_metrics = [svm_report['1'][metric] for metric in metrics]
rf_metrics = [rf_report['1'][metric] for metric in metrics]

# Plot Precision, Recall, F1-Score comparison
fig, ax = plt.subplots(figsize=(8, 6))
bar_width = 0.2
x = range(len(metrics))

ax.bar([p - bar_width for p in x], logreg_metrics, width=bar_width, label='Logistic Regression', color='blue')
ax.bar(x, svm_metrics, width=bar_width, label='SVM', color='green')
ax.bar([p + bar_width for p in x], rf_metrics, width=bar_width, label='Random Forest', color='red')

ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_title('Model Comparison: Precision, Recall, F1-Score')
ax.set_ylabel('Score')
ax.legend()

plt.show()
