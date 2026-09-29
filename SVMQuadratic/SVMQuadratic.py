import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

np.random.seed(42)

X = np.random.uniform(-5, 5, (300, 2))

y = (X[:, 1] > X[:, 0]**2 - 4).astype(int)

model = SVC(kernel="rbf", C=1.0, gamma="scale")
model.fit(X, y)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 500),
    np.linspace(y_min, y_max, 500)
)

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 7))
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k")
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("SVM Classification with Quadratic Distribution")
plt.show()

print("Accuracy:", model.score(X, y))