import matplotlib.pyplot as plt
import sys

import cv2
from scipy import misc

import cs_utils

# the portion of the original block coefficients to use in basis fourier reconstruction
# the portion of the most influential frequencies to use in discrete fourier transform reconstruction
RETAIN_RATIO = 1.0

# square block size (width and height) to chunk up the image by for faster processing
BLOCK_SIZE = 64

# size to rescale the input images to
IMG_RESCALE_SIZE = (256, 256)


def process_images(image_paths):

    """
    Creates a best focus reconstruction from a set of focal stack images.

    :param image_paths: list of path to focal stack image files
    :type image_paths: list
    """

    # load images as grayscale and rescale the images
    sys.stdout.write('Loading images:\n\r')
    sys.stdout.flush()

    images = []
    for path in image_paths:
        sys.stdout.write('Loading image: {} ...'.format(path))
        sys.stdout.flush()
        img = misc.imresize(cv2.imread(path, 0), IMG_RESCALE_SIZE)
        images.append(img)
        sys.stdout.write(' Done\n\r')
        sys.stdout.flush()

    # create a best focused reconstruction from focal stack images
    sys.stdout.write('Creating a best focused reconstruction from focal stack ...')
    sys.stdout.flush()
    rec_all_focused = cs_utils.reconstruct_bf_focalstack_gray(images, BLOCK_SIZE, RETAIN_RATIO)
    sys.stdout.write(' Done\n\r')
    sys.stdout.flush()

    # save the results
    sys.stdout.write('Saving the results ...\n\r')
    sys.stdout.flush()
    img_name = image_paths[0].split('/')[-1].split('.')[-2]
    save_path_af = 'results/{}_all_focused_{}.jpg'.format(img_name, int(RETAIN_RATIO * 100))
    cv2.imwrite(save_path_af, rec_all_focused)
    sys.stdout.write('Saved into {}\n\r'.format(save_path_af))
    sys.stdout.flush()

    # create a figure and display images
    sys.stdout.write('Displaying the results ...')
    sys.stdout.flush()

    plt.figure(1)

    for i in range(min(3, len(images))):
        plt.subplot(int('22{}'.format(i+1)))
        plt.imshow(images[i], cmap="gray")
        plt.title("Image {}".format(i))

    plt.subplot(224)
    plt.imshow(rec_all_focused, cmap="gray")
    plt.title("Reconstruction with top %d%% of coefficients" % (RETAIN_RATIO * 100))

    plt.show()

    sys.stdout.write(' Done\n\r')
    sys.stdout.flush()


def main(argv):
    """
    :param argv: list of command line arguments
    """
    if len(argv) == 1:
        print "Not enough arguments\nUsage: python reconstruct_fs.py <path to focal stack image files>"
        return
    else:
        image_paths = argv[1:]
        process_images(image_paths)


if __name__ == '__main__':
    main(sys.argv)
