import math

data = [
    (40, 20, "Red"),
    (50, 50, "Blue"),
    (60, 90, "Blue"),
    (10, 25, "Red"),
    (70, 70, "Blue"),
    (60, 10, "Red"),
    (25, 80, "Blue")
]

brightness = float(input("Enter new brightness: "))
saturation = float(input("Enter new saturation: "))

distances = []

for b, s, c in data:
    distance = math.sqrt((brightness - b) ** 2 + (saturation - s) ** 2)
    distances.append((distance, b, s, c))

distances.sort()

print("\nEuclidean Distances:")

for distance, b, s, c in distances:
    print(f"({b}, {s}) -> {distance:.2f} -> {c}")