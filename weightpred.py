import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Read dataset
df = pd.read_csv("bodyweight.csv")

# Display first rows
print(df.head())

# Check dataset size
print(df.shape)

# Check null values
print(df.isnull().sum())

# Convert Gender text into numbers
# Male = 1 , Female = 0
df["Gender"] = df["Gender"].map({
    "Male": 1,
    "Female": 0
})

# Independent variables
X = df[["Height", "Age", "Gender"]]

# Dependent variable
y = df["Weight"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict values
y_pred = model.predict(X_test)

print("Predicted Values")
print(y_pred)

# Error calculation
error = mean_absolute_error(y_test, y_pred)

print("Error =", error)

# Graph
plt.scatter(df["Height"], df["Weight"])

plt.title("Body Weight Prediction")

plt.xlabel("Height")

plt.ylabel("Weight")

plt.show()