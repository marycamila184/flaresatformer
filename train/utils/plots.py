from matplotlib import pyplot as plt

def show_sample(dataset, idx=0, title="Sample"):
    img, mask = dataset[idx]   

    img = img.numpy()
    mask = mask.numpy()[0]
    band = 0

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(img[band], cmap="gray")
    plt.title(f"{title} - Image (band {band})")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="hot")
    plt.title(f"{title} - Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()