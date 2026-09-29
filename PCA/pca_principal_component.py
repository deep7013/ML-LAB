import numpy as np

variables = [
    "Climate",
    "Housing",
    "Health",
    "Crime",
    "Transportation",
    "Education",
    "Arts",
    "Recreation",
    "Economy"
]

loadings = np.array([
    [0.190, 0.017, 0.207],
    [0.544, 0.020, 0.204],
    [0.782, -0.605, 0.144],
    [0.365, 0.294, 0.585],
    [0.585, 0.085, 0.234],
    [0.394, -0.273, 0.027],
    [0.985, 0.126, -0.111],
    [0.520, 0.402, 0.519],
    [0.142, 0.150, 0.239]
])

for i in range(loadings.shape[1]):
    index = np.argmax(np.abs(loadings[:, i]))
    print(f"PC{i + 1}: {variables[index]} ({loadings[index, i]})")

index = np.unravel_index(np.argmax(np.abs(loadings)), loadings.shape)

print("\nStrongest overall variable:")
print(f"{variables[index[0]]} -> PC{index[1] + 1} ({loadings[index]})")