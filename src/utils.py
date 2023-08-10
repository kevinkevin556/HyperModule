import matplotlib.pyplot as plt
from torch.utils.data import Subset
from torchvision.transforms.functional import to_pil_image


def plot_images(image_seq, xlabs, cmap, figsize=6):
    n = len(image_seq)
    figsize = (figsize, figsize * n // 2)
    fig, ax = plt.subplots(n, len(xlabs), gridspec_kw={"wspace": 0, "hspace": 0}, figsize=figsize)
    for row, images in enumerate(image_seq):
        for col, (img, xlab) in enumerate(zip(images, xlabs)):
            if xlab == "Image":
                ax[row][col].imshow(to_pil_image(img), cmap="gray")
            else:
                ax[row][col].imshow(to_pil_image(img), cmap=cmap)
            ax[row][col].set_xlabel(xlab)
            ax[row][col].tick_params(bottom=False, left=False, labelbottom=False, labelleft=False  )
    plt.show()


def get_dataset(dataloader):
    dataset = dataloader.dataset
    if isinstance(dataset, Subset):
        dataset = dataset.dataset
    return dataset