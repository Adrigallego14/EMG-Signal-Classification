import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

def evaluate_knn(X_train, y_train, X_test, y_test, n_neighbors=4):
    # Training the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    
    # Predictions
    knn_pred = knn.predict(X_test)
    
    # Evaluation metrics
    knn_accuracy = accuracy_score(y_test, knn_pred)
    knn_f1 = f1_score(y_test, knn_pred, average='weighted')
    knn_precision = precision_score(y_test, knn_pred, average='weighted')
    knn_recall = recall_score(y_test, knn_pred, average='weighted')
    conf_matrix_knn = confusion_matrix(y_test, knn_pred)
    
    # Print evaluation metrics
    print('K-Nearest Neighbors (KNN):')
    print('Accuracy:', "%.2f" % (knn_accuracy*100))
    print('F1:', "%.2f" % (knn_f1*100))
    print('Precision:', "%.2f" % (knn_precision*100))
    print('Recall:', "%.2f" % (knn_recall*100))
    
    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_knn, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predictions')
    plt.ylabel('True Values')
    plt.title('Confusion Matrix')
    plt.show()

# Usage example:
# evaluate_knn(X_train, y_train, X_test, y_test, n_neighbors=4)
