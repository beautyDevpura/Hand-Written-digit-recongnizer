import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import fetch_openml
import pickle

mnist = fetch_openml('mnist_784')
X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target,random_state=49)

print(X_train.shape)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

KNN = KNeighborsClassifier(n_neighbors=6)
KNN.fit(X_train, y_train)

print(KNN.score(X_train, y_train))
print(KNN.score(X_test, y_test))

pickle.dump(KNN, open('model.pkl', 'wb'))

#Resources: https://github.com/nachi-hebbar/Streamlit-ML-Web-App/commit/fe202431b8d78211e7dd2022a62ffeaef49c60ed
