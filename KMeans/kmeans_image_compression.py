import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

image_path = input("Enter image path: ")
k = int(input("Enter number of colors: "))

image = Image.open(image_path).convert("RGB")
image = image.resize((396, 396))

pixels = np.array(image)
pixels = pixels.reshape(-1, 3)

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
kmeans.fit(pixels)

compressed_pixels = kmeans.cluster_centers_[kmeans.labels_]
compressed_pixels = np.clip(compressed_pixels, 0, 255).astype(np.uint8)

compressed_image = compressed_pixels.reshape(396, 396, 3)

Image.fromarray(compressed_image).save("KMeans/compressed_image.png")

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(np.array(image))
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(compressed_image)
plt.title(f"Compressed Image - {k} Colors")
plt.axis("off")

plt.show()

print("Original shape:", np.array(image).shape)
print("Compressed shape:", compressed_image.shape)
print("Compressed image saved as compressed_image.png")