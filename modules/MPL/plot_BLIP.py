import numpy as np
import matplotlib.pyplot as plt

def plot_result(image, clusters, captions):
    unique_clusters = np.unique(clusters)
    cmap = plt.cm.get_cmap('tab20', len(unique_clusters))  # 'tab20' is a good colormap for categorical data
    # Create a plot with a colorbar that has labels
    fig, axs = plt.subplots(1, 2, figsize=(25, 7))  # 1 row, 2 columns

    axs[0].imshow(image.squeeze().permute(1, 2, 0))
    axs[0].set_title('Image')
    # The first subplot will display your raw image
    cax = axs[1].imshow(clusters.squeeze())
    axs[1].set_title('MaskBLIP')
    # This creates a colorbar for the segmentation plot
    cbar = fig.colorbar(cax, ax=axs[0], ticks=unique_clusters, spacing='proportional')
    # This sets the labels of the colorbar to correspond to your captions
    cbar.ax.set_yticklabels(captions)  # change fontsize and rotation as necessary

    # Show the plot
    plt.tight_layout()
    plt.show()