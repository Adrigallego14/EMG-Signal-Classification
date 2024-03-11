import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.svm import SVC

def evaluate_rbf_svm(X_train, y_train, X_test, y_test):
    # Training the SVM with RBF kernel
    rbf = SVC(kernel='rbf', gamma='scale', C=5).fit(X_train, y_train)
    
    # Predictions
    rbf_pred = rbf.predict(X_test)
    
    # Evaluation metrics
    rbf_accuracy = accuracy_score(y_test, rbf_pred)
    rbf_f1 = f1_score(y_test, rbf_pred, average='weighted')
    rbf_precision = precision_score(y_test, rbf_pred, average='weighted')
    rbf_recall = recall_score(y_test, rbf_pred, average='weighted')
    conf_matrx_svm = confusion_matrix(y_test, rbf_pred)
    
    # Print evaluation metrics
    print('Accuracy (RBF Kernel): ', "%.2f" % (rbf_accuracy*100))
    print('F1 (RBF Kernel): ', "%.2f" % (rbf_f1*100))
    print('Precision (RBF Kernel): ', "%.2f" % (rbf_precision*100))
    print('Recall (RBF Kernel): ', "%.2f" % (rbf_recall*100))
    
    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrx_svm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predictions')
    plt.ylabel('True Values')
    plt.title('Confusion Matrix')
    plt.show()
