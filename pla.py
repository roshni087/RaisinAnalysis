# perceptron learning algorithm model
# load dataset
import pandas as pd
data = pd.read_csv("raisin.csv")

# preprocess data
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

label_encoder = LabelEncoder()
data['Class'] = label_encoder.fit_transform(data['Class'])

imputer = SimpleImputer(strategy='mean')
data.fillna(data.mean(), inplace=True)

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data.iloc[:, :-1])

scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data_scaled)

# split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_normalized[:, :-1], data['Class'], test_size=0.2, random_state=42)

# train the model
from sklearn.linear_model import Perceptron
perceptron_model = Perceptron()
perceptron_model.fit(X_train, y_train)

# test the model
perceptron_test_accuracy = perceptron_model.score(X_test, y_test)
print(f"Perceptron Test Accuracy: {perceptron_test_accuracy}\n")


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# compute decision function scores
decision_train = perceptron_model.decision_function(X_train)
decision_test = perceptron_model.decision_function(X_test)

# compute roc curve and auc for training set
fpr_train, tpr_train, _ = roc_curve(y_train, decision_train)
roc_auc_train = auc(fpr_train, tpr_train)

# compute roc curve and auc for test set
fpr_test, tpr_test, _ = roc_curve(y_test, decision_test)
roc_auc_test = auc(fpr_test, tpr_test)

# plot roc curves
plt.figure(figsize=(8, 6))
plt.plot(fpr_train, tpr_train, color='darkorange', lw=2, label=f'Train ROC curve (AUC = {roc_auc_train:.2f})')
plt.plot(fpr_test, tpr_test, color='cornflowerblue', lw=2, label=f'Test ROC curve (AUC = {roc_auc_test:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# compute accuracy on training set
train_accuracy = perceptron_model.score(X_train, y_train)
print(f"Train Accuracy: {train_accuracy}")

# compute accuracy on test set
test_accuracy = perceptron_model.score(X_test, y_test)
print(f"Test Accuracy: {test_accuracy}")

# check for significant difference in accuracies
if train_accuracy - test_accuracy > 0.05:
    print("The model might be overfitting.")
else:
    print("The model is not overfitting.")
