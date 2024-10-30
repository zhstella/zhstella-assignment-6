import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

X = np.random.rand(10, 1)
Y = np.random.normal(0, 1, (10, 1))
model = LinearRegression()
model.fit(X, Y)
slope = model.coef_[0]
intercept = model.intercept_
plt.figure(figsize=(10, 5))
plt.scatter(X, Y, color="blue", label="Data")
plt.plot(X, model.predict(X), color="red", label="Regression Line")
plt.xlabel("X")
plt.ylabel("Y")
    # plt.title(f"Linear Regression: Y = {slope:.2f}X + {intercept:.2f}")
plt.legend()
    # Add a title showing the regression line equation using the slope and intercept values.
    # Finally, save the plot to "static/plot1.png" using plt.savefig()
plt.show()
plt.close()
import pdb ; pdb.set_trace()