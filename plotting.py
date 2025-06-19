##########################
# DO NOT CHANGE THIS FILE!
##########################

from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_images(og, up_nn, up_bl, upfactor, down_nn, down_bl, downfactor):
    fig, axs = plt.subplots(2, 3)
    fig.suptitle('Bitmap Scaling')
    axs[0][0].imshow(og)
    axs[0][0].set_title('Original')

    axs[0][1].imshow(Image.fromarray(up_nn))
    axs[0][1].set_title('Nearest Neighbor')

    axs[0][2].imshow(Image.fromarray(up_bl))
    axs[0][2].set_title('Bilinear')

    axs[1][0].imshow(og)

    axs[1][1].imshow(Image.fromarray(down_nn))

    axs[1][2].imshow(Image.fromarray(down_bl))

    axs[0][0].set_ylabel(f'Scaling {round(upfactor, 2)}', fontsize=14)
    axs[1][0].set_ylabel(f'Scaling {round(downfactor, 2)}', fontsize=14)
    for row in axs:
        for ax in row:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    fig.tight_layout()
    plt.savefig('results.png')
    plt.show()