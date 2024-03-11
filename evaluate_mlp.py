from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

def evaluate_mlp(X_train, y_train, X_test, y_test, hidden_layer_sizes=(200,), max_iter=1000):
    # Training the MLP classifier
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter)
    mlp.fit(X_train, y_train)
    
    # Predictions
    mlp_pred = mlp.predict(X_test)
    
    # Evaluation metrics
    mlp_accuracy = accuracy_score(y_test, mlp_pred)
    mlp_f1 = f1_score(y_test, mlp_pred, average='weighted')
    mlp_precision = precision_score(y_test, mlp_pred, average='weighted')
    mlp_recall = recall_score(y_test, mlp_pred, average='weighted')
    conf_matrix_mlp = confusion_matrix(y_test, mlp_pred)
    
    # Print evaluation metrics
    print('Multi-layer Perceptron (Neural Network):')
    print('Accuracy:', "%.2f" % (mlp_accuracy*100))
    print('F1:', "%.2f" % (mlp_f1*100))
    print('Precision:', "%.2f" % (mlp_precision*100))
    print('Recall:', "%.2f" % (mlp_recall*100))
    
    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_mlp, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predictions')
    plt.ylabel('True Values')
    plt.title('Confusion Matrix')
    plt.show()

# Usage example:
# evaluate_mlp(X_train, y_train, X_test, y_test, hidden_layer_sizes=(200,), max_iter=1000)
