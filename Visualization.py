import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml


mnist = fetch_openml('mnist_784')

X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target,
                                                    random_state=49)

print('X_train shape', X_train.shape)
print('y_train shape', y_train.shape)

print(X_train.iloc[1].shape)

print(X_train.iloc[1]) # Flatten 28*28 pixel representation of an element 1, 28*28 = 784 vectors

Real_img = X_train.iloc[1].values.reshape(28, 28) # reshape to 28*28 for image representation
plt.imshow(Real_img)
plt.title(y_train.iloc[1], fontsize=15)

Pxl_img = X_train.iloc[1].values.reshape(28, 28)

plt.imshow(Pxl_img, cmap='gray')
plt.xticks(np.arange(0,28))
plt.yticks(np.arange(0,28))
plt.title(y_train[1], fontsize=15)
plt.grid()
plt.show()

Pxl = X_train.iloc[1].values.reshape(28, 28)
imgToPxl = pd.DataFrame(Pxl) # pixel representation of Image as Data frame # Only for Visualization purpose
print(imgToPxl)


plt.figure(figsize=(15, 9))

plt.subplot(2, 3, 1)
img1 = X_train.iloc[10].values.reshape(28, 28)
plt.imshow(img1, cmap='gray') # converting color image to gray
plt.title(y_train.iloc[10], fontsize=15)

plt.subplot(2, 3, 2)
img2 = X_train.iloc[100].values.reshape(28, 28)
plt.imshow(img2, cmap='gray')
plt.title(y_train.iloc[100], fontsize=15);

plt.subplot(2, 3, 3)
img3 = X_train.iloc[1000].values.reshape(28, 28)
plt.imshow(img3, cmap='gray')
plt.title(y_train.iloc[1000], fontsize=15);

plt.subplot(2, 3, 4)
img4 = X_train.iloc[5000].values.reshape(28, 28)
plt.imshow(img4, cmap='gray')
plt.title(y_train.iloc[5000], fontsize=15);

plt.subplot(2, 3, 5)
img5 = X_train.iloc[10000].values.reshape(28, 28)
plt.imshow(img5, cmap='gray')
plt.title(y_train.iloc[10000], fontsize=15);

plt.subplot(2, 3, 6)
img6 = X_train.iloc[50000].values.reshape(28, 28)
plt.imshow(img6, cmap='gray')
plt.title(y_train.iloc[50000], fontsize=15);

plt.show()


