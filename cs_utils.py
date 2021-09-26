import math
import sys

import cv2
import numpy as np

import comp_sense.Sketching as Sketch


def magnitude_spectrum_gray(img):
    """
    Creates the magnitude spectrum image (visual) of the provided grayscale image

    :param img: grayscale image
    :type img: numpy.ndarray

    :return: grayscale magnitude spectrum image
    :rtype: numpy.ndarray
    """

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    return magnitude_spectrum


def basis_fourier(block, num_retain_coeff):
    """
    Creates coefficients retaining top k coefficients of the block image,
     and creates a proper mixing matrix.

    :param block: block image
    :type block: numpy.ndarray
    :param num_retain_coeff: the number of top coefficients to retain
    :type num_retain_coeff: int

    :return: mixing matrix, coefficients
    """

    # convert the image to a single column vector
    img_vector = block.ravel()

    # compute the FFT
    fourier = np.fft.fft(img_vector)

    # retain the top 'k' coefficients, and zero out the remaining ones
    sorted_indices = np.argsort(-1.0 * np.absolute(fourier).ravel())
    coefficients = fourier
    coefficients[sorted_indices[num_retain_coeff:]] = 0.0
    coefficients = np.asmatrix(coefficients).T

    # generate basis matrix for these coefficients
    basis = Sketch.computeFourierBasis(len(coefficients))

    return basis, coefficients


def basis_fourier_fs(block_focal_stack, num_retain_coeff):
    """
    Creates best focus coefficients from set of coefficients created retaining top k coefficients of the block image,
     and creates a proper mixing matrix.

    :param block_focal_stack: list of focal stack block images
    :type block_focal_stack: list
    :param num_retain_coeff: the number of top coefficients to retain
    :type num_retain_coeff: int

    :return: mixing matrix, coefficients
    """

    # create coefficients list containing coefficients of each focal stack block image
    coefficients_list = []
    for block in block_focal_stack:
        # convert the image to a single column vector
        img_vector = block.ravel()

        # compute the FFT
        fourier = np.fft.fft(img_vector)

        # retain the top 'k' coefficients, and zero out the remaining ones
        sorted_indices = np.argsort(-1.0 * np.absolute(fourier).ravel())
        coefficients = fourier
        coefficients[sorted_indices[num_retain_coeff:]] = 0.0
        coefficients_list.append(coefficients)

    # create best focused coefficients by selecting most extremes of the block coefficients for each index
    coefficients = coefficients_list[0]
    for i in range(len(coefficients)):
        coeff_set = []
        for j in range(len(coefficients_list)):
            coeff_set.append(coefficients_list[j][i])
        extreme_index = np.argsort(-1.0 * np.absolute(coeff_set).ravel())[0]
        coefficients[i] = coefficients_list[extreme_index][i]

    coefficients = np.asmatrix(coefficients).T

    # generate basis matrix for these coefficients
    basis = Sketch.computeFourierBasis(len(coefficients))

    return basis, coefficients


def reconstruct_bf_gray(img, block_size, coeff_retain_ratio):
    """
    Creates a basis fourier reconstruction of the image.

    :param img: a grayscale image to reconstruct a new image from
    :type img: numpy.ndarray
    :param block_size: square block size (width and height) to chunk up the image by,
    image width and height must be divisible by block_size
    :type block_size: int
    :param coeff_retain_ratio: portion of the top coefficients to retain
    :type coeff_retain_ratio: float

    :return: reconstructed image
    :rtype: numpy.ndarray
    """

    # create block segments from the image
    block_shape = (block_size, block_size)
    blocks = Sketch.getBlocks(img, block_shape[0])
    num_blocks = len(blocks)

    # reconstruct new blocks
    sys.stdout.write('Performing blockwise basis fourier reconstruction on image:\n\r')
    sys.stdout.write('image size: {}\n\rblock size: {}\n\rcoeff_retain_ratio: {}%\n\rnumber of blocks: {}\n\r'
                     .format(img.shape, block_shape, int(coeff_retain_ratio * 100), num_blocks))
    sys.stdout.flush()

    num_retain_coeff = int(block_shape[0] * block_shape[1] * coeff_retain_ratio)
    rec_blocks = []
    for i in range(num_blocks):
        sys.stdout.write('\rReconstructing block {} of {} ...'.format(i+1, num_blocks))
        sys.stdout.flush()
        basis, coefficients = basis_fourier(blocks[i], num_retain_coeff)
        rec = (basis * coefficients).reshape(block_shape)
        rec = np.absolute(rec)
        rec_blocks.append(rec)

    sys.stdout.write(' Done \n\r')
    sys.stdout.flush()

    # assemble the reconstructed blocks into a new image
    sys.stdout.write('Assempling reconstructed blocks ...')
    sys.stdout.flush()
    reconstruction = Sketch.assembleBlocks(rec_blocks, img.shape)
    sys.stdout.write(' Done\n\r')
    sys.stdout.flush()

    return reconstruction


def reconstruct_bf_focalstack_gray(images, block_size, coeff_retain_ratio):
    """
    Creates a best focused basis fourier reconstruction from the focal stack images.

    :param images: a list of grayscale focal stack images to reconstruct a new best focused image from
    :type images: list
    :param block_size: square block size (width and height) to chunk up the images by,
    image width and height must be divisible by block_size
    :type block_size: int
    :param coeff_retain_ratio: portion of the top coefficients to retain
    :type coeff_retain_ratio: float

    :return: reconstructed image
    :rtype: numpy.ndarray
    """

    # create block segments from the focal stack images
    block_shape = (block_size, block_size)
    block_lists = []
    for img in images:
        blocks = Sketch.getBlocks(img, block_shape[0])
        block_lists.append(blocks)
    num_images = len(images)
    num_focal_stacks = len(block_lists[0])
    num_blocks = num_focal_stacks * num_images

    # reconstruct new blocks
    sys.stdout.write('Performing blockwise basis fourier reconstruction on focal stack images:\n\r')
    sys.stdout.write('image size: {}\n\rblock size: {}\n\rcoeff_retain_ratio: {}%\n\rnumber of images : {}\n\r'
                     .format(images[0].shape, block_shape, int(coeff_retain_ratio * 100), num_images))
    sys.stdout.write('number of blocks: {}\n\rnumber of focal stacks: {}\n\r'
                     .format(num_blocks, num_focal_stacks))
    sys.stdout.flush()

    num_retain_coeff = int(block_shape[0] * block_shape[1] * coeff_retain_ratio)
    rec_blocks = []
    for i in range(num_focal_stacks):
        block_focal_stack = []
        for j in range(len(block_lists)):
            block_focal_stack.append(block_lists[j][i])
        sys.stdout.write('\rReconstructing block focal stack {} of {} (stack size: {} blocks) ...'
                         .format(i + 1, num_focal_stacks, len(block_focal_stack)))
        sys.stdout.flush()
        basis, coefficients = basis_fourier_fs(block_focal_stack, num_retain_coeff)
        rec = (basis * coefficients).reshape(block_shape)
        rec = np.absolute(rec)
        rec_blocks.append(rec)

    sys.stdout.write(' Done \n\r')
    sys.stdout.flush()

    # assemble the reconstructed blocks into a new image
    sys.stdout.write('Assempling reconstructed blocks ...')
    sys.stdout.flush()
    reconstruction = Sketch.assembleBlocks(rec_blocks, images[0].shape)
    sys.stdout.write(' Done\n\r')
    sys.stdout.flush()

    return reconstruction


def reconstruct_dft_gray(img, signal_retain_ratio):
    """
    Creates a discrete fourier transform reconstruction of the image.

    :param img: a grayscale image to reconstruct a new image from
    :type img: numpy.ndarray
    :param signal_retain_ratio: portion of the most influential frequencies to retain
    :type signal_retain_ratio: float

    :return: reconstructed image
    :rtype: numpy.ndarray
    """

    # compute the DFT
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # get image shape and center
    rows, cols = img.shape
    crow, ccol = rows / 2, cols / 2

    # create a circular mask to retain signal_retain_ratio of most influential frequencies
    mask_radius = int(math.sqrt(img.shape[0] * img.shape[1] * signal_retain_ratio / math.pi))
    mask = np.zeros((rows, cols, 2), np.uint8)
    cv2.circle(mask, (crow, ccol), mask_radius, (1, 1), -1)

    # apply mask
    fshift = dft_shift * mask

    # inverse DFT
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

    return img_back
