import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = [[6], [8], [10], [14],   [18]]
y = [[7], [9], [13], [17.5], [18]]

'''
plt.figure()
plt.xlabel("Pizza in Inches")
plt.ylabel("Price in Dollars")
plt.title("Prices of Pizza")
plt.plot(X,y,'k.')
'''


model = LinearRegression()
model.fit(X,y)

plt.axis([0, 25, 0, 25])
plt.grid(True)
plt.show()
print("The cost of 20 inch pizza is $ %f " %model.predict(20))
print("Residual Sum  = %f"%np.mean(model.predict(X)-y)**2)
