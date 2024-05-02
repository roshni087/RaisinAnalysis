# logistic regression
#load dataset
import pandas as pd
data = pd.read_csv("raisin.csv")

#preprocess data
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

#split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_normalized[:, :-1], data['Class'], test_size=0.2, random_state=42)

#train the model
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

#test the model
lr_test_accuracy = lr_model.score(X_test, y_test)
print(f"Logistic Regression Test Accuracy: {lr_test_accuracy}")


