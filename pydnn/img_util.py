import tools
import preprocess as pp

import numpy as np


def dimensions_to_fit_images(length):
    sqrt = int(np.sqrt(length))
    rows = sqrt + 1 if sqrt * sqrt < length else sqrt
    cols = sqrt + 1 if sqrt * rows < length else sqrt
    return rows, cols


def show_image_tiles(images, canvas_dims=None):
    import scipy.misc

    canvas_dims = (dimensions_to_fit_images(images.shape[0]) if
                   canvas_dims is None else canvas_dims)
    image = tools.tile_2d_images(images, canvas_dims)
    scipy.misc.imshow(image)


def show_images_as_tiles(images, size, canvas_dims=None):
    images_array = np.zeros(shape=(len(images), size[0], size[1]),
                               dtype='uint8')
    for array_slice, image in zip(images_array, images):
        # print(image.shape, image)
        resized = pp._resize_preserving_aspect_ratio(image, size)
        # print(array_slice.shape, image.shape, resized.shape, size)
        # print(array_slice)
        array_slice[:] = resized
    show_image_tiles(images_array, canvas_dims)


def show_images(images, titles=None, canvas_dims=None):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import scipy

    titles = [''] * len(images) if titles is None else titles
    rows, cols = (dimensions_to_fit_images(images.shape[0]) if
                  canvas_dims is None else canvas_dims)

    fig = plt.figure()
    for n, (title, image) in enumerate(zip(images, titles)):
        scipy.misc.imshow(image)
        sub = fig.add_subplot(rows, cols, n + 1)
        plt.imshow(image, interpolation='none', cmap=cm.Greys_r)
        sub.set_title('{} ({}x{})'.format(title, image.shape[0], image.shape[1]),
                      size=10)
        sub.axis('off')
    plt.show()


def test_show_images():
    rng = np.random.RandomState()
    import scipy
    from os.path import basename
    images = []
    for i in range(30):
        f = train_set.get_random_file(rng)
        image = scipy.misc.imread(f)
        title = '{} {} {}x{}'.format(i, basename(f), image.shape[0], image.shape[1])
        images.append((title, image))
    show_images(images, (pp._resize_preserving_aspect_ratio(image, 128, 128)))
# test_show_images()