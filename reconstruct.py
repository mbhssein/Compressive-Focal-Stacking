import matplotlib.pyplot as plt
import sys

import cv2
from scipy import misc
import cs_utils

# the portion of the original block coefficients to use in basis fourier reconstruction
# the portion of the most influential frequencies to use in discrete fourier transform reconstruction
RETAIN_RATIO = 0.75

# square block size (width and height) to chunk up the image by for faster processing
BLOCK_SIZE = 64

# size to rescale the input images to
IMG_RESCALE_SIZE = (256, 256)


def process_images(image_paths):
    """
    Creates reconstructions of images using basis fourier and discrete fourier transform methods.

    :param image_paths: list of path to image files to reconstruct using compressed sensing
    :type image_paths: list
    """

    # process all images
    for path in image_paths:

        # load image as grayscale and rescale the image
        sys.stdout.write('Loading image: {} ...'.format(path))
        sys.stdout.flush()

        img = misc.imresize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), IMG_RESCALE_SIZE)

        sys.stdout.write(' Done\n\r')
        sys.stdout.flush()

        # calculate magnitude spectrum (for visualization only)
        sys.stdout.write('Calculating magnitude spectrum ...'.format(path))
        sys.stdout.flush()
        magnitude_spectrum = cs_utils.magnitude_spectrum_gray(img)
        sys.stdout.write(' Done\n\r')
        sys.stdout.flush()

        # reconstruct the image using basis fourier method
        sys.stdout.write('Creating a BF reconstruction ...\n\r')
        sys.stdout.flush()
        reconstruction_bf = cs_utils.reconstruct_bf_gray(img, BLOCK_SIZE, RETAIN_RATIO)
        sys.stdout.write('Done creating a BF reconstruction\n\r')
        sys.stdout.flush()

        # reconstruct the image using discrete fourier transform method
        sys.stdout.write('Creating a DFT reconstruction ...\n\r')
        sys.stdout.flush()
        reconstruction_dft = cs_utils.reconstruct_dft_gray(img, RETAIN_RATIO)
        sys.stdout.write('Done creating a DFT reconstruction\n\r')
        sys.stdout.flush()

        # save the results
        sys.stdout.write('Saving the results ...\n\r')
        sys.stdout.flush()
        img_name = path.split('/')[-1].split('.')[-2]

        save_path_mag_spec = 'results/{}_mag_spec.jpg'.format(img_name)
        cv2.imwrite(save_path_mag_spec, magnitude_spectrum)
        sys.stdout.write('Saved into {}\n\r'.format(save_path_mag_spec))
        sys.stdout.flush()

        save_path_bf = 'results/{}_rec_bf_{}.jpg'.format(img_name, int(RETAIN_RATIO * 100))
        cv2.imwrite(save_path_bf, reconstruction_bf)
        sys.stdout.write('Saved into {}\n\r'.format(save_path_bf))
        sys.stdout.flush()

        save_path_dft = 'results/{}_rec_dft_{}.jpg'.format(img_name, int(RETAIN_RATIO * 100))
        cv2.imwrite(save_path_dft, reconstruction_dft)
        sys.stdout.write('Saved into {}\n\r'.format(save_path_dft))
        sys.stdout.flush()

        # create a figure and display images
        sys.stdout.write('Displaying the results ...')
        sys.stdout.flush()

        plt.figure(1)

        plt.subplot(221)
        plt.imshow(img, cmap="gray")
        plt.title("Original Image")

        plt.subplot(222)
        plt.imshow(magnitude_spectrum, cmap="gray")
        plt.title("Magnitude Spectrum")

        plt.subplot(223)
        plt.imshow(reconstruction_bf, cmap="gray")
        plt.title("BF Reconstruction with top %d%% of coefficients" % (RETAIN_RATIO * 100))

        plt.subplot(224)
        plt.imshow(reconstruction_dft, cmap="gray")
        plt.title("DFT Reconstruction with most influential %d%% of frequencies" % (RETAIN_RATIO * 100))

        plt.show()

        sys.stdout.write(' Done\n\r')
        sys.stdout.flush()


def main(argv):
    """
    :param argv: list of command line arguments
    """
    if len(argv) == 1:
        print "Not enough arguments\nUsage: python reconstruct.py <path to image files>"
        return
    else:
        image_paths = argv[1:]
        process_images(image_paths)


if __name__ == '__main__':
    main(sys.argv)
