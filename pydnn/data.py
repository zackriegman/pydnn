import tools

from os.path import join
import numpy as np
from scipy.misc import imread
import theano


class DirectoryLabeledImageSet(object):
    """
    Builds training data from a directory where each subdirectory
    is the name of a class, and contains all the examples of images that class.

    :param string base_dir: the directory containing the class directories
    :param string dtype: the data type to use for the ndarray containing the labels
    """
    def __init__(self, base_dir, dtype=theano.config.floatX):
        """
        Build data set from images in directories by class names.
        """
        print('DirectoryLabeledImageSet: {}'.format(base_dir))
        self.base_dir = base_dir
        self.dtype = dtype

    def get_labels(self):
        return sorted(tools.get_sub_dirs(self.base_dir, rel=True))

    def get_files(self, num_files=None):
        files = [(fn, label) for label in self.get_labels() for fn in
                sorted(tools.get_files(join(self.base_dir, label), rel=True))]
        if num_files is None:
            return files
        else:
            import random
            random.seed(12345)
            return random.sample(files, num_files)

    def get_random_file(self, rng):
        sub_dir = rng.choice(tools.get_sub_dirs(self.base_dir, cache=True))
        return rng.choice(tools.get_files(sub_dir, cache=True))

    def build(self, num_images=None):
        files = self.get_files(num_images)
        num_images = len(files)
        images = [None] * num_images
        classes = np.zeros(shape=num_images, dtype=self.dtype)
        file_indices = np.zeros(shape=num_images, dtype='int')
        labels = self.get_labels()

        print('loading {} of 30336 images...'.format(num_images))
        dotter = tools.Dot(skip=num_images / 20)
        for i, (fn, label) in enumerate(files):
            images[i] = imread(join(self.base_dir, label, fn))
            classes[i] = labels.index(label)
            file_indices[i] = i
            dotter.dot(str(i) + ' ')
        dotter.stop()
        return images, classes, file_indices


class UnlabeledImageSet(object):
    """
    Builds an inference set from a directory containing unlabeled images.

    :param string base_dir: the directory containing the images
    """
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.files = None

    def get_files(self):
        if self.files is None:
            self.files = sorted(tools.get_files(self.base_dir, rel=True))
        return self.files

    def build(self, start, stop):
        files = self.get_files()[start: stop]
        return files, [imread(join(self.base_dir, f)) for f in files]