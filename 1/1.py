# %%
import pandas as pd
import numpy as np
# %%
from sklearn.datasets import load_digits
digits=load_digits()
import matplotlib.pyplot as plt
print(digits.data.shape)
print(digits.target.shape)
plt.figure(figsize=(20,4))
for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
 plt.subplot(1, 5, index + 1)
 plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)
 plt.title('Training: %i\n' % label, fontsize = 20)
 from sklearn.model_selection import train_test_split
 x_train,x_test,y_train,y_test=train_test_split(digits.data,digits.target,test_size=0.25,random_state=0)
 from sklearn.linear_model import LogisticRegression
 lr=LogisticRegression()
lr.fit(x_train,y_train)
lr.predict(x_test[0].reshape(1,-1))
lr.predict(x_test[0:10])