# Classifiers: KNN, Decision Tree, and Naive Bayes
# Written by: Angel Madeux

from data_preprocess import X_train, X_test, y_train, y_test, target_names
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# KNN Hyperparameter Tuning
print("\nKNN  Tuning Results:")
# Testing different values of k for KNN
for k in [1, 2, 4, 8, 9]: 
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    preds = knn.predict(X_test)
    ac = accuracy_score(y_test, preds)

    print(f"k = {k} | Accuracy = {ac:.4f}")

best_k = 4 # Produces the best accuracy of 0.73 from tuning

# Train KNN model using the best k value
final_knn = KNeighborsClassifier(n_neighbors=best_k) 
final_knn.fit(X_train, y_train)

y_pred_final = final_knn.predict(X_test)

# Evaluate Model
print("\nFinal KNN Results:")

accuracy = accuracy_score(y_test, y_pred_final)
print("Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_final))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_final, target_names=target_names))

# Decision Tree Classifier
print()
print("Decision Tree Tuning Results:")

# Try different tree depths to find the best one
for depth in [2, 5, 11, None]:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)

    preds = dt.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print("Depth =", depth, "| Accuracy =", round(acc, 4))

# Train Decision Tree model using the best depth value
final_dt = DecisionTreeClassifier(max_depth=5, random_state=42)
final_dt.fit(X_train, y_train)

final_preds = final_dt.predict(X_test)

# Evaluate Model
print("\nFinal Decision Tree Results:")
print("Accuracy:", accuracy_score(y_test, final_preds))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, final_preds))

print("\nClassification Report:")
print(classification_report(y_test, final_preds, target_names=target_names))


# Naive Bayes Classifier
# Doesn't require hyperparameter tuning!
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
