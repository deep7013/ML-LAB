import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Read dataset
df = pd.read_csv("taxi.csv")

# Display first rows
print(df.head())

# Shape of dataset
print(df.shape)

# Check null values
print(df.isnull().sum())

# Independent and dependent variables
X = df[["trip_seconds", "trip_miles"]]
y = df["fare"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Build model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

print("Predicted values")
print(y_pred)

# Error
error = mean_absolute_error(y_test, y_pred)

print("Error =", error)

# Plot graph
plt.scatter(df["trip_miles"], df["fare"])

plt.plot(df["trip_miles"], model.predict(X), color="red")

plt.title("Taxi Fare Prediction")

plt.xlabel("Trip Miles")

plt.ylabel("Fare")

plt.show()