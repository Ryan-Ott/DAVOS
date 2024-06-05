import os
import sys
import torch

import matplotlib.pyplot as plt

def main():
    if len(sys.argv) < 2:
        print("Please provide the tensor path as a command line argument.")
        return

    tensor_path = sys.argv[1]
    if not os.path.isfile(tensor_path):
        print("Invalid tensor path.")
        return

    tensor = torch.load(tensor_path)

    if tensor.dim() == 3 and tensor.size(0) == 3:  # Check if it’s a 3-channel image tensor
        # Create a figure with three subplots, stacked vertically
        fig, axes = plt.subplots(3, 1, figsize=(5, 15))  # Adjust the size as needed

        for i, (ax, cls) in enumerate(zip(axes, ['background', 'chair', 'person'])):
            # Extract the ith channel and convert it to a numpy array
            channel = tensor[i, :, :].numpy()

            # Plot the channel as a binary mask on the corresponding subplot
            ax.imshow(channel, cmap='gray')  # Use grayscale color map
            ax.set_title(f'Channel {i+1} ({cls})')
            ax.axis('off')  # Hide the axis

        plt.tight_layout()

        # Save the plot in the same directory as the tensor with the filename "plot.png"
        plot_path = os.path.join(os.path.dirname(tensor_path), "plot.png")
        plt.savefig(plot_path)
    else:
        print("Tensor details:", tensor)

if __name__ == "__main__":
    main()