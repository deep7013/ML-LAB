from collections import Counter

predictions = [
    "Apple",
    "Apple",
    "Banana",
    "Apple",
    "Banana",
    "Apple",
    "Apple",
    "Banana",
    "Apple",
    "Apple"
]

votes = Counter(predictions)

print("Decision Tree Predictions:")
for i, prediction in enumerate(predictions, 1):
    print(f"Tree-{i}: {prediction}")

print("\nVotes:")
for fruit, count in votes.items():
    print(f"{fruit}: {count}")

final_class = votes.most_common(1)[0][0]

print("\nFinal Class:", final_class)
print("Fruit likely to be taken often:", final_class)