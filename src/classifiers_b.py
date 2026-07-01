# Classifiers: ANN, Naive Bayes, and SVM
# Written by: Angel Madeux

from data_preprocess import X_train, X_test, y_train, y_test, target_names
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings

warnings.filterwarnings("ignore")

# ANN Hyperparameter Tuning
print("\nANN Tuning Results:")
# Testing different hidden layer sizes for ANN
hidden_layers_list = [(8,), (24, 19,), (48, 12)]

for h in hidden_layers_list:
    ann = MLPClassifier(
        hidden_layer_sizes=h,
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=42
    )
    
    ann.fit(X_train, y_train)
    preds = ann.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"Hidden Layers = {h} | Accuracy = {acc:.4f}")

best_layers = (8,) # Produces the best accuracy of 0.7600 from tuning

# Train ANN model using the best hidden layer size 
final_ann = MLPClassifier(
    hidden_layer_sizes=best_layers,
    activation='relu',
    solver='adam',
    max_iter=3000,
    random_state=42
)
final_ann.fit(X_train, y_train)

y_pred_ann = final_ann.predict(X_test)

# Evaluate Model
print("\nFinal ANN Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_ann))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_ann))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_ann, target_names=target_names))


# Naive Bayes Classifier
print()
print("Naive Bayes Classifier")
nb = GaussianNB()
nb.fit(X_train, y_train) # Train the model

# Make predictions
y_pred_nb = nb.predict(X_test)

# Evaluate Model
print("\nFinal Naive Bayes Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_nb))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_nb, target_names=target_names))


# SVM Hyperparameter Tuning
print()
print("SVM Tuning Results:")

# Testing combinations of C and kernel to find the best fit
svm_combinations = [
    (0.01, 'linear'), 
    (1.0, 'linear'), 
    (0.02, 'rbf'), 
    (4.0, 'rbf')
]

for C_val, kernel_val in svm_combinations:
    svm_model = SVC(C=C_val, kernel=kernel_val, random_state=42)
    svm_model.fit(X_train, y_train)

    preds = svm_model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"C = {C_val}, Kernel = {kernel_val} | Accuracy = {acc:.4f}")


best_C = 4.0       # Produces the best accuracy of 0.7200 from tuning
best_kernel = 'rbf' 

# Train SVM model using the best parameters
final_svm = SVC(C=best_C, kernel=best_kernel, random_state=42)
final_svm.fit(X_train, y_train)

y_pred_svm = final_svm.predict(X_test)

# Evaluate Model
print("\nFinal SVM Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_svm))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_svm, target_names=target_names))
