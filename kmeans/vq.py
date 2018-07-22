from PIL import Image

from kmeans import kmeans
import numpy as np
import matplotlib.pyplot as plt


def vq(k):
    img = Image.open('data/sample1.jpg')
    img.load()
    img = np.asarray(img, dtype="int")
    fea = img.astype(float).reshape(-1, 3)

    # YOUR (TWO LINE) CODE HERE
    idx, ctrs, _ = kmeans(fea, k)
    fea = ctrs[idx]

    plt.figure(figsize=(10, 5))
    ax = plt.subplot("121")
    ax.set_title("Origin")
    plt.imshow(img.astype(int))

    ax = plt.subplot("122")
    ax.set_title("Compressed (K = %d)" % k)
    plt.imshow(fea.reshape(img.shape).astype(int))
