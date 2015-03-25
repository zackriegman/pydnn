__author__ = 'isaac'

import scipy
import scipy.ndimage
import numpy as np
import tools
import theano


def _flatten_3d_to_2d(data):
    s = data.shape
    return data.reshape((s[0], s[1] * s[2]))


def split_training_data(data, batch_size, train_per, valid_per, test_per):
    """
    Split training data into training set, validation set and test sets.  If split
    results in incomplete batches this function will allocate observations from
    incomplete batches in validation and test sets to the training set to attempt
    to make a complete batch.  This function also reports on the split and whether
    there were observations that did not fit evenly into any batch.  (Currently
    :class:`NN <pydnn.nn.NN>` assumes a validation and test set, so if allocating 100% of data to
    training set, :func:`split_training_data` will duplicate the first batch of
    the training set for the validation and test sets so :class:`NN <pydnn.nn.NN>` does not fail.
    In that case, obviously the validation and test loss will be meaningless.)

    :param tuple data: all the data, including x and y values
    :param int batch_size: number of observations that will be in each batch
    :param float train_per: percentage of data to put in training set
    :param float valid_per: percentage of data to put in validation set
    :param float test_per: percentage of data to put in test set
    :return: a list containing the training, validation and test sets, each
        of which is a list containing each variable (in the case where x data is a single
        image that will be a list of images and a list of y classes, but there can also
        be more than one x variable).
    """
    assert train_per + valid_per + test_per == 100
    num_obs = len(data[0])

    # calculate number of observations in set, rounding down to batch size
    def calc_num(per):
        num = int(num_obs * per / 100.0)
        num -= num % batch_size
        return num

    num_train = calc_num(train_per)
    num_valid = calc_num(valid_per)
    num_test = calc_num(test_per)

    # if there are enough residual observation to make a batch, add them to train
    num_train += (num_obs - num_train - num_valid - num_test) / batch_size * batch_size

    # print out some sanity check numbers
    num_discard = num_obs - num_train - num_valid - num_test
    print("train: {} obs, {} batches; valid: {} obs, {} batches; test: {} obs, {} batches".
          format(num_train, num_train / batch_size, num_valid, num_valid / batch_size,
                 num_test, num_test / batch_size))
    print("of {} observations, {} are used and {} are discarded because they do not fit evenly into a batch of size {}".
          format(num_obs, num_obs - num_discard, num_discard, batch_size))

    # I use a separate random seed here to make sure that this shuffle happens
    # the same way each time; and won't get inadvertently change if I change the order
    # of code execution at some point.  That way, if I don't change the percentage
    # breakdown between train, validation and test sets I should always be getting
    # the same sets.
    _shuffle_lists(np.random.RandomState(24680), data)

    # builds sets, e.g. set[test|valid|train][images|shapes|classes]
    if train_per == 100:  # have to add some dummy data to valid/test to avoid exceptions
        print("TRAINING ON ALL DATA, VALIDATION AND TESTING SETS ARE ONLY DUMMIES")
        sets = [[var[start:stop] for var in data]
                for (start, stop)
                in ((0, num_train),
                    (0, batch_size),
                    (0, batch_size))]
    else:
        sets = [[var[start:stop] for var in data]
                for (start, stop)
                in ((0, num_train),
                    (num_train, num_train + num_valid),
                    (num_train + num_valid, num_train + num_valid + num_test))]

    # if show_samples:
    #     import scipy
    #     from MyTools import tile_2d_images
    #
    #     for s in sets:
    #         image = tile_2d_images(s[0], (batch_size / 20 + 1, 20))
    #         scipy.misc.imshow(image)

    return sets


def _expand_white(image, size):
    """
    Given an image and a target size, creates a new image of the target size and
    places image in the middle, thus expanding the image with white.

    :param ndarray image: image being expanded
    :param tuple size: size of the new image
    :return: white expanded image
    """
    white = np.ndarray(size, dtype=image.dtype)
    white.fill(255)
    # white.fill(0)
    v_off = int((size[0] - image.shape[0]) / 2)
    h_off = int((size[1] - image.shape[1]) / 2)
    white[v_off:v_off + image.shape[0],
          h_off:h_off + image.shape[1]] = image
    return white


class Resizer(object):
    """
    Base class for :class:`StretchResizer`, :class:`ContiguousBoxPreserveAspectRatioResizer`,
    :class:`ContiguousBoxStretchResizer`, :class:`ThresholdBoxPreserveAspectRatioResizer`,
    :class:`ThresholdBoxStretchResizer`, :class:`PreserveAspectRatioResizer`,
    and :class:`StochasticStretchResizer`

    Some resizers may want to resize training images differently from
    validation or testing images so this class gives them the option of doing so.
    """
    def training(self, image, size):
        return self._resize(image, size)

    def prediction(self, image, size):
        return self._resize(image, size)


class StretchResizer(Resizer):
    """
    Stretches the images to a uniform shape ignoring aspect ratio
    """
    @staticmethod
    def _resize(image, size):
        return scipy.misc.imresize(image, size)


def _crop_with_threshold(image, threshold):
    def top():
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i, j] < threshold:
                    return i

    def bottom():
        for i in range(1, image.shape[0]):
            for j in range(image.shape[1]):
                if image[-i, j] < threshold:
                    return image.shape[0] - i

    def left():
        for i in range(image.shape[1]):
            for j in range(image.shape[0]):
                if image[j, i] < threshold:
                    return i

    def right():
        for i in range(1, image.shape[1]):
            for j in range(image.shape[0]):
                if image[j, -i] < threshold:
                    return image.shape[1] - i

    lft, rght, tp, bttm = left(), right(), top(), bottom()
    if (any(num is None for num in (lft, rght, tp, bttm)) or
        lft >= rght or tp >= bttm):
        return image
    else:
        return image[tp: bttm, lft: rght]


def _crop_biggest_contiguous(image, threshold):
    # contiguous region selection based on:
    # http://stackoverflow.com/questions/4087919/how-can-i-improve-my-paw-detection
    thresh = 255 - image > threshold
    coded, num = scipy.ndimage.label(thresh)
    slices = [image[slc] for slc in scipy.ndimage.find_objects(coded)]
    biggest = slices[np.argmax([slc.size for slc in slices])]
    return biggest


class ContiguousBoxPreserveAspectRatioResizer(Resizer):
    """
    First crops the images around the largest contiguous region, then stretches
    them to a uniform size preserving aspect ratio.
    """
    def __init__(self, threshold):
        self.threshold = threshold

    def _resize(self, image, size):
        image = _crop_biggest_contiguous(image, self.threshold)
        return _resize_preserving_aspect_ratio(image, size)

class ContiguousBoxStretchResizer(Resizer):
    """
    First crops the images around the largest contiguous region, then stretches
    them to a uniform ignoring aspect ratio.
    """
    def __init__(self, threshold):
        self.threshold = threshold

    def _resize(self, image, size):
        image = _crop_biggest_contiguous(image, self.threshold)
        return scipy.misc.imresize(image, size)


class ThresholdBoxPreserveAspectRatioResizer(Resizer):
    """
    First crops the images, throwing away outside space without pixels
    that exceed a given threshold, then stretches
    them to a uniform size preserving aspect ratio.
    """
    def __init__(self, threshold):
        self.threshold = threshold

    def _resize(self, image, size):
        image = _crop_with_threshold(image, self.threshold)
        return _resize_preserving_aspect_ratio(image, size)


class ThresholdBoxStretchResizer(Resizer):
    """
    First crops the images, throwing away outside space without pixels
    that exceed a given threshold, then stretches
    them to a uniform ignoring aspect ratio.
    """
    def __init__(self, threshold):
        self.threshold = threshold

    def _resize(self, image, size):
        image = _crop_with_threshold(image, self.threshold)
        return scipy.misc.imresize(image, size)


def _resize_preserving_aspect_ratio(image, size):
    if float(image.shape[0]) / image.shape[1] > float(size[0]) / size[1]:
            new_size = (size[0], int((float(size[0]) / image.shape[0]) * image.shape[1]))
    else:
        new_size = (int((float(size[1]) / image.shape[1]) * image.shape[0]), size[1])
    image = scipy.misc.imresize(image, new_size)
    image = _expand_white(image, size)

    return image

class PreserveAspectRatioResizer(Resizer):
    """
    Stretches images to a uniform size preserving aspect ratio.
    """
    @staticmethod
    def _resize(image, size):
        return _resize_preserving_aspect_ratio(image, size)


class StochasticStretchResizer(Resizer):
    """
    Stretches images to a uniform size ignoring aspect ratio.

    Randomly varies how much training images are stretched.
    """
    def __init__(self, rng, rand_range):
        self.rng = rng
        self.rand_range = rand_range

    def _crop_range(self, rand, fixed):
        if rand > fixed:
            off = (rand - fixed) / 2
            return off, off + fixed
        else:
            return 0, fixed

    def training(self, image, size):
        # randomly fluctuate each dimension by normal dist with variance 0.05
        # using the normal distribution can cause exceptions due to extreme values
        # rand = numpy.cast['int'](size * (.05 * self.rng.randn(2) + 1))

        # randomly fluctuate each dimension by uniform distribution
        rand = np.cast['int'](size * (self.rng.uniform(self.rand_range[0],
                                         self.rand_range[0], size=2)))
        resized = scipy.misc.imresize(image, rand)

        # if dimension is bigger than desired size, crop to desired size
        vert = self._crop_range(rand[0], size[0])
        horz = self._crop_range(rand[1], size[1])
        cropped = resized[vert[0]:vert[1], horz[0]:horz[1]]

        # if dimension is smaller than desired size, fill with white
        return _expand_white(cropped, size)

    def prediction(self, image, size):
        # print( 'image', image, 'size', size)
        return scipy.misc.imresize(image, size)


def _make_shape_array(images, dtype):
    shapes = np.zeros(shape=(len(images), 2), dtype=dtype)
    for image, shape in zip(images, shapes):
        shape[:] = image.shape
    return shapes


class Rotator360(object):
    """
    Rotates training set images randomly.

    (Also zero centers by pixel, normalizes, shuffles, resizes, etc.)

    :param data: x and y data
    :param image_shape: the target image shape
    :param resizer: the resizer to use when uniformly resizing images
    :param rng: random number generator
    :param string dtype: the datatype to output
    """
    def __init__(self, data, image_shape, resizer, rng, dtype=theano.config.floatX):

        # start_time = time.time()
        self.dtype = dtype
        self.rng = rng

        self.train_image = None
        self.train_shape = None
        self.train_class = None
        self.valid_image = None
        self.valid_shape = None
        self.valid_class = None
        self.test_image = None
        self.test_shape = None
        self.test_class = None
        self.data = None
        self.set_data(data)

        self.image_shape = image_shape
        self.resizer = resizer

        self.image_mean = self._approximate_image_mean(self.train_image)
        self.shape_mean = np.mean(self.train_shape, axis=0)
        self.shape_std = np.std(self.train_shape, axis=0)
        self.shape_range = (np.max(self.train_shape, axis=0) -
                            np.min(self.train_shape, axis=0))
        print('shape', 'mean', self.shape_mean, 'std', self.shape_std,
              'range', self.shape_range)

    def set_data(self, data):
        dtype = self.dtype
        self.train_image = data[0][0]
        self.train_shape = _make_shape_array(data[0][0], dtype)
        self.train_class = data[0][1].astype(dtype)
        self.valid_image = data[1][0]
        self.valid_shape = _make_shape_array(data[1][0], dtype)
        self.valid_class = data[1][1].astype(dtype)
        self.test_image = data[2][0]
        self.test_shape = _make_shape_array(data[2][0], dtype)
        self.test_class = data[2][1].astype(dtype)

        self.data = data

        # I want an even distribution of random rotations, but if there is any
        # regularity in the orientations (e.g. because of gravity) I want to break
        # that up, so each image gets a fixed rotation, and all the images are adjusted
        # by the master rotation after each epoch, so that after 360 epochs, the net
        # has seen each in each of 360 degree rotations.
        self.image_rots = self.rng.choice(
                np.arange(0, 360, 1),
                size=len(self.train_image) + len(self.valid_image) + len(self.test_image),
                replace=True)
        self.master_rots = np.concatenate([
            self.rng.choice(
                np.arange(offset, 360 + offset, 1),
                size=360,
                replace=False) for offset in (0, 0.5, 0.25, 0.75)])
        self.rot_index = 0

    def _approximate_image_mean(self, training_images):
        images = self._resize_consolidate(training_images, self.resizer.training)
        images = self._transform_images(images)
        return np.mean(images, axis=0)

    def _resize_consolidate(self, images, resizer):
        # convert a list of images of varying sizes into a single
        # array of images of uniform size
        images_array = np.zeros(shape=(len(images), self.image_shape[0],
                                          self.image_shape[1]),
                                   dtype='uint8')
        for array_slice, image in zip(images_array, images):

            array_slice[:] = resizer(image, self.image_shape)

        return images_array

    def _zero_center_normalize(self, images, shapes):
        # normalize images and shapes
        images = (images - self.image_mean) / 255.0

        shapes = (shapes - self.shape_mean) / self.shape_std
        # shapes = (shapes - self.shape_mean) / self.shape_range
        # shapes = (shapes - self.shape_mean) / self.shape_mean
        cast = np.cast[self.dtype]
        return cast(_flatten_3d_to_2d(images)), shapes

    def _transform_images(self, images):
        # rotations seem to change the background color to grey haphazardly
        # when done on floating point type images so assert that this is
        # 256 valued int.
        assert images.dtype == 'uint8'
        images = images.copy()
        rotations = self.image_rots + self.master_rots[self.rot_index]
        self.rot_index = self.rot_index + 1 if self.rot_index < self.master_rots.size - 1 else 0

        for index, image in enumerate(images):
            rot = rotations[index] if rotations[index] < 360 else rotations[index] - 360
            image[:] = scipy.ndimage.interpolation.rotate(
                input=image,
                angle=rot,
                reshape=False,
                mode='constant',
                cval=255)
            if self.rng.choice([True, False]):
                image[:] = np.fliplr(image)
        return images

    def _prep_for_training(self, images, shapes, classes):
        # transformation must happen before normalization because rotations,
        # shearing, etc, introduce white values at the corners
        # which must also be zero centered and normalized

        images = self._resize_consolidate(images, self.resizer.training)
        images = self._transform_images(images)
        images, shapes = self._zero_center_normalize(images, shapes)

        # must copy the classes because they'll be shuffled inplace
        # (images and shapes lists are already copied in above functions)
        classes = classes.copy()
        _shuffle_lists(self.rng, [images, shapes, classes])

        return self._wrap_dict(images, shapes, classes)


    def _prep_for_prediction(self, images, shapes):
        images = self._resize_consolidate(images, self.resizer.prediction)
        return self._zero_center_normalize(images, shapes)

    def _wrap_dict(self, images, shapes, classes=None):
        if classes is None:
            return {'images': images, 'shapes': shapes}
        else:
            return {'images': images, 'shapes': shapes}, classes

    def shape_for(self, channel):
        img_shape = self.image_shape
        shapes = {'images': (1, img_shape[0], img_shape[1]), 'shapes': (2,)}
        return shapes[channel]

    @tools.time_once
    def get_train(self):
        return self._prep_for_training(self.train_image, self.train_shape,
                                       self.train_class)

    def get_all_train(self):
        x = self.train_image + self.valid_image + self.test_image
        shapes = np.vstack((self.train_shape, self.valid_shape, self.test_shape))
        y = np.append(self.train_class, np.append(self.valid_class, self.test_class))
        return self._prep_for_training(x, shapes, y)

    def get_valid(self):
        images, shapes = self._prep_for_prediction(self.valid_image, self.valid_shape)
        # print('get_valid', images, shapes)
        return self._wrap_dict(images, shapes, self.valid_class)

    def get_test(self):
        images, shapes = self._prep_for_prediction(self.test_image, self.test_shape)
        # print('get_test', images, shapes)
        return self._wrap_dict(images, shapes, self.test_class)

    def get_raw_x(self):
        return [{'images': s[0]} for s in self.data]

    def get_raw_y(self):
        return [s[1] for s in self.data]

    def get_files(self):
        return [s[2] for s in self.data]

    def preprocess(self, data):
        image_data = data['images']
        shape_data = _make_shape_array(image_data, self.dtype)
        images, shapes = self._prep_for_prediction(image_data, shape_data)
        return self._wrap_dict(images, shapes)

    def __getstate__(self):
        return {'image_mean': self.image_mean,
                'shape_mean': self.shape_mean,
                'shape_std': self.shape_std,
                'shape_range': self.shape_range,
                'image_shape': self.image_shape,
                'resizer': self.resizer,
                'dtype': self.dtype,
                'rng': self.rng,
                'image_rots': self.image_rots,
                'master_rots': self.master_rots,
                'rot_index': self.rot_index
        }


class Rotator360PlusGeometry(object):
    """
    Rotates training set images randomly, but also generates
    additional geometric data about the size and orientation of
    the organism in the image.

    (Also zero centers by pixel, normalizes, shuffles, resizes, etc.)

    :param data: x and y data
    :param image_shape: the target image shape
    :param resizer: the resizer to use when uniformly resizing images
    :param rng: random number generator
    :param string dtype: the datatype to output
    """
    def __init__(self, data, image_shape, resizer, rng, dtype):

        # start_time = time.time()
        self.dtype = dtype
        self.rng = rng

        self.train_image = None
        self.train_geom = None
        self.train_class = None
        self.valid_image = None
        self.valid_geom = None
        self.valid_class = None
        self.test_image = None
        self.test_geom = None
        self.test_class = None
        self.data = None
        self.set_data(data)

        self.image_shape = image_shape
        self.resizer = resizer

        self.image_mean = self._approximate_image_mean(self.train_image)
        self.geom_mean = np.mean(self.train_geom, axis=0)
        self.geom_std = np.std(self.train_geom, axis=0)
        self.geom_range = (np.max(self.train_geom, axis=0) -
                            np.min(self.train_geom, axis=0))
        print('geom', 'mean', self.geom_mean, 'std', self.geom_std,
              'range', self.geom_range)

    def make_geometric_data(self, images):
        geoms = np.zeros(shape=(len(images), 2 + 2 + 1 + 2), dtype=self.dtype)
        print('calculating geometric data...')
        dotter = tools.Dot(skip=len(images)/40)
        for image, geom in zip(images, geoms):
            img, rot, ud, lr = _canonicalize_image(image, include_info_in_result=True)
            geom[0:2] = image.shape
            geom[2:4] = _crop_biggest_contiguous(img, 2).shape
            geom[4] = rot
            geom[5] = ud
            geom[6] = lr
            dotter.dot()
        dotter.stop()
        return geoms

    def set_data(self, data):
        dtype = self.dtype
        self.train_image = data[0][0]
        self.train_geom = self.make_geometric_data(data[0][0])
        self.train_class = data[0][1].astype(dtype)
        self.valid_image = data[1][0]
        self.valid_geom = self.make_geometric_data(data[1][0])
        self.valid_class = data[1][1].astype(dtype)
        self.test_image = data[2][0]
        self.test_geom = self.make_geometric_data(data[2][0])
        self.test_class = data[2][1].astype(dtype)

        self.data = data

        # I want an even distribution of random rotations, but if there is any
        # regularity in the orientations (e.g. because of gravity) I want to break
        # that up, so each image gets a fixed rotation, and all the images are adjusted
        # by the master rotation after each epoch, so that after 360 epochs, the net
        # has seen each in each of 360 degree rotations.
        self.image_rots = self.rng.choice(
                np.arange(0, 360, 1),
                size=len(self.train_image) + len(self.valid_image) + len(self.test_image),
                replace=True)
        self.master_rots = np.concatenate([
            self.rng.choice(
                np.arange(offset, 360 + offset, 1),
                size=360,
                replace=False) for offset in (0, 0.5, 0.25, 0.75)])
        self.rot_index = 0

    def _approximate_image_mean(self, training_images):
        images = self._resize_consolidate(training_images, self.resizer.training)
        images = self._transform_images(images)
        return np.mean(images, axis=0)

    def _resize_consolidate(self, images, resizer):
        # convert a list of images of varying sizes into a single
        # array of images of uniform size
        images_array = np.zeros(shape=(len(images), self.image_shape[0],
                                       self.image_shape[1]),
                                dtype='uint8')
        for array_slice, image in zip(images_array, images):
            array_slice[:] = resizer(image, self.image_shape)

        return images_array

    def _zero_center_normalize(self, images, geoms):
        # normalize images and shapes
        images = (images - self.image_mean) / 255.0

        geoms = (geoms - self.geom_mean) / self.geom_std
        # shapes = (shapes - self.shape_mean) / self.shape_range
        # shapes = (shapes - self.shape_mean) / self.shape_mean
        cast = np.cast[self.dtype]
        return cast(_flatten_3d_to_2d(images)), geoms

    def _transform_images(self, images):
        # rotations seem to change the background color to grey haphazardly
        # when done on floating point type images so assert that this is
        # 256 valued int.
        assert images.dtype == 'uint8'
        images = images.copy()
        rotations = self.image_rots + self.master_rots[self.rot_index]
        self.rot_index = self.rot_index + 1 if self.rot_index < self.master_rots.size - 1 else 0

        for index, image in enumerate(images):
            rot = rotations[index] if rotations[index] < 360 else rotations[index] - 360
            image[:] = scipy.ndimage.interpolation.rotate(
                input=image,
                angle=rot,
                reshape=False,
                mode='constant',
                cval=255)
            if self.rng.choice([True, False]):
                image[:] = np.fliplr(image)
        return images

    def _prep_for_training(self, images, geoms, classes):
        # transformation must happen before normalization because rotations,
        # shearing, etc, introduce white values at the corners
        # which must also be zero centered and normalized

        images = self._resize_consolidate(images, self.resizer.training)
        images = self._transform_images(images)
        images, geoms = self._zero_center_normalize(images, geoms)

        # must copy the classes because they'll be shuffled inplace
        # (images and shapes lists are already copied in above functions)
        classes = classes.copy()
        _shuffle_lists(self.rng, [images, geoms, classes])

        return self._wrap_dict(images, geoms, classes)

    def _prep_for_prediction(self, images, geoms):
        images = self._resize_consolidate(images, self.resizer.prediction)
        return self._zero_center_normalize(images, geoms)

    def _wrap_dict(self, images, geoms, classes=None):
        if classes is None:
            return {'images': images, 'geometry': geoms}
        else:
            return {'images': images, 'geometry': geoms}, classes

    def shape_for(self, channel):
        img_shape = self.image_shape
        shapes = {'images': (1, img_shape[0], img_shape[1]), 'geometry': (7,)}
        return shapes[channel]

    @tools.time_once
    def get_train(self):
        return self._prep_for_training(self.train_image, self.train_geom,
                                       self.train_class)

    def get_all_train(self):
        x = self.train_image + self.valid_image + self.test_image
        geoms = np.vstack((self.train_geom, self.valid_geom, self.test_geom))
        y = np.append(self.train_class, np.append(self.valid_class, self.test_class))
        return self._prep_for_training(x, geoms, y)

    def get_valid(self):
        images, geom = self._prep_for_prediction(self.valid_image, self.valid_geom)
        # print('get_valid', images, shapes)
        return self._wrap_dict(images, geom, self.valid_class)

    def get_test(self):
        images, geom = self._prep_for_prediction(self.test_image, self.test_geom)
        # print('get_test', images, shapes)
        return self._wrap_dict(images, geom, self.test_class)

    def get_raw_x(self):
        return [{'images': s[0]} for s in self.data]

    def get_raw_y(self):
        return [s[1] for s in self.data]

    def get_files(self):
        return [s[2] for s in self.data]

    def preprocess(self, data):
        image_data = data['images']
        geom_data = self.make_geometric_data(image_data)
        images, geoms = self._prep_for_prediction(image_data, geom_data)
        return self._wrap_dict(images, geoms)

    def __getstate__(self):
        return {'image_mean': self.image_mean,
                'geom_mean': self.geom_mean,
                'geom_std': self.geom_std,
                'geom_range': self.geom_range,
                'image_shape': self.image_shape,
                'resizer': self.resizer,
                'dtype': self.dtype,
                'rng': self.rng,
                'image_rots': self.image_rots,
                'master_rots': self.master_rots,
                'rot_index': self.rot_index
        }


class Canonicalizer(object):
    """
    Rotates and flips all images into a canonicalized form.  Using
    a statistical measure of object height rotates each image to
    minimize height.  (Can also either (1) flip images so aggregate
    pixel intensity is highest in one corner, or (2) generate random
    flips of training images.)

    (Also zero centers by pixel, normalizes, shuffles, resizes, etc.)

    :param data: x and y data
    :param image_shape: the target image shape
    :param resizer: the resizer to use when uniformly resizing images
    :param rng: random number generator
    :param string dtype: the datatype to output
    """
    def __init__(self, data, image_shape, resizer, rng, dtype):
        self.dtype = dtype
        self.rng = rng
        self.image_shape = image_shape
        self.resizer = resizer

        self.train_image = None
        self.train_geom = None
        self.train_class = None
        self.valid_image = None
        self.valid_geom = None
        self.valid_class = None
        self.test_image = None
        self.test_geom = None
        self.test_class = None

        self.image_mean = None
        self.geom_mean = None
        self.geom_std = None
        self.geom_range = None

        self.data = None
        self.set_data(data)

    def set_data(self, data):
        dtype = self.dtype

        images, geoms = self._canonicalize_images(data[0][0],
                                                  self.resizer.training)
        self.geom_mean = np.mean(geoms, axis=0)
        self.geom_std = np.std(geoms, axis=0)
        self.geom_range = (np.max(geoms, axis=0) -
                           np.min(geoms, axis=0))
        print('shape', 'mean', self.geom_mean, 'std', self.geom_std,
              'range', self.geom_range)
        self.train_image = images
        self.image_mean = np.mean([image for image in self.train_image] +
                                  [np.fliplr(image) for image in self.train_image] +
                                  [np.flipud(image) for image in self.train_image] +
                                  [np.fliplr(np.flipud(image)) for image in self.train_image],
                                     axis=0)
        self.train_geom = self._zero_center_normalize_geoms(geoms)
        self.train_class = data[0][1].astype(dtype)

        images, geoms = self._canonicalize_images(data[1][0],
                                                  self.resizer.training)
        self.valid_image = images
        self.valid_geom = self._zero_center_normalize_geoms(geoms)
        self.valid_class = data[1][1].astype(dtype)

        images, geoms = self._canonicalize_images(data[2][0],
                                                  self.resizer.training)
        self.test_image = images
        self.test_geom = self._zero_center_normalize_geoms(geoms)
        self.test_class = data[2][1].astype(dtype)

        # all the above should result in copies so the data below
        # is completely untransformed
        self.data = data

    def _zero_center_normalize_images(self, images):
        images = (images - self.image_mean) / 255.0
        cast = np.cast[self.dtype]
        return cast(_flatten_3d_to_2d(images))

    def _zero_center_normalize_geoms(self, geoms):
        geoms = (geoms - self.geom_mean) / self.geom_std
        return geoms

    def _canonicalize_images(self, images, resizer):
        images_array = np.zeros(shape=(len(images), self.image_shape[0],
                                       self.image_shape[1]),
                                dtype='uint8')
        geoms = np.zeros(shape=(len(images), 2 + 2 + 1 + 2), dtype=self.dtype)

        print('canonicalizing {} images'.format(len(images)))
        dotter = tools.Dot(skip=len(images) / 20)
        for i, (img_arr, geom, image) in enumerate(zip(images_array, geoms, images)):
            img, rot, ud, lr = _canonicalize_image(image,
                                                  include_info_in_result=True)
            img_arr[:] = resizer(img, self.image_shape)
            geom[0:2] = image.shape
            geom[2:4] = _crop_biggest_contiguous(img, 2).shape
            geom[4] = rot
            geom[5] = ud
            geom[6] = lr
            dotter.dot(str(i) + ' ')
        dotter.stop()
        return images_array, geoms

    def _wrap_dict(self, images, geoms, classes=None):
        if classes is None:
            return {'images': images, 'geometry': geoms}
        else:
            return {'images': images, 'geometry': geoms}, classes

    def shape_for(self, channel):
        img_shape = self.image_shape
        shapes = {'images': (1, img_shape[0], img_shape[1]), 'geometry': (7,)}
        return shapes[channel]

    @tools.time_once
    def _prep_for_training(self, images, geoms, classes):
        images, geoms, classes = images.copy(), geoms.copy(), classes.copy()

        for image in images:
            if self.rng.choice([True, False]):
                image[:] = np.fliplr(image)
            if self.rng.choice([True, False]):
                image[:] = np.flipud(image)

        _shuffle_lists(self.rng, [images,
                                 geoms,
                                 classes])
        # import plankton
        # plankton.show_image_tiles(images)
        return self._wrap_dict(self._zero_center_normalize_images(images), geoms, classes)

    @tools.time_once
    def get_train(self):
        return self._prep_for_training(self.train_image, self.train_geom,
                                       self.train_class)

    def get_all_train(self):
        x = np.vstack((self.train_image, self.valid_image, self.test_image))
        shapes = np.vstack((self.train_geom, self.valid_geom, self.test_geom))
        y = np.append(self.train_class, np.append(self.valid_class, self.test_class))
        return self._prep_for_training(x, shapes, y)

    def get_valid(self):
        return self._wrap_dict(self._zero_center_normalize_images(self.valid_image),
                               self.valid_geom, self.valid_class)

    def get_test(self):
        return self._wrap_dict(self._zero_center_normalize_images(self.test_image),
                               self.test_geom, self.test_class)

    def preprocess(self, data):
        image_data = data['images']
        images, geoms = self._canonicalize_images(image_data, self.resizer.prediction)
        return self._wrap_dict(self._zero_center_normalize_images(images),
                               self._zero_center_normalize_geoms(geoms))

    def get_raw_x(self):
        return [{'images': s[0]} for s in self.data]

    def get_raw_y(self):
        return [s[1] for s in self.data]

    def get_files(self):
        return [s[2] for s in self.data]

    def __getstate__(self):
        return {'image_mean': self.image_mean,
                'geom_mean': self.geom_mean,
                'geom_std': self.geom_std,
                'geom_range': self.geom_range,
                'image_shape': self.image_shape,
                'resizer': self.resizer,
                'dtype': self.dtype,
                'rng': self.rng}


def _canonicalize_image(img, include_info_in_result=False, do_flips=False):
    def stat_height(image):
        image = 255 - image
        row_sums = np.sum(image, axis=1)
        all_sum = np.sum(row_sums)

        def find_middle():
            run_tot = 0
            for j, row in enumerate(row_sums):
                run_tot += row
                if run_tot > all_sum / 2:
                    return j
        mid = find_middle()
        height = 0
        for i, sum in enumerate(row_sums):
            height += (abs(i - mid) ** 2) * sum
        return height

    def rotation(image, deg):
        return scipy.ndimage.interpolation.rotate(
            input=image,
            angle=deg,
            reshape=False,
            mode='constant',
            cval=255)

    def minimize(func, image, deg=(0.0, 180.0)):
        diff = deg[1] - deg[0]
        if diff < 1:
            return deg[0] + diff / 2

        step = diff / 3.0
        rots = deg[0] + step, deg[0] + 2.0 * step
        imgs = [rotation(image, rot) for rot in rots]
        scores = [func(img) for img in imgs]
        # scipy.misc.imshow(numpy.column_stack(imgs))
        range = (deg[0], deg[0] + diff / 2.0) if scores[0] < scores[1] \
            else (deg[0] + diff / 2.0, deg[1])
        return minimize(func, image, range)

    def flips(image):
        h_height = image.shape[0] / 2
        h_width = image.shape[1] / 2

        return (np.sum(image[:h_height]) < np.sum(image[h_height:]),
                np.sum(image[:, :h_width]) > np.sum(image[:, h_width:]))

    # rotations seem to change the background color to grey haphazardly
    # when done on floating point type images so assert that this is
    # 256 valued int.
    assert img.dtype == 'uint8'

    dim = max(img.shape)
    img = _expand_white(img, (dim, dim))
    rot = minimize(stat_height, img)
    img = rotation(img, rot)

    # don't do flips here if also doing them randomly above for training...
    # they make the pixel mean invalid for the validation/test/prediction sets
    if do_flips:
        ud, lr = flips(img)
        img = np.flipud(img) if ud else img
        img = np.fliplr(img) if lr else img
    if include_info_in_result:
        ud, lr = flips(img)
        return img, rot, ud, lr
    else:
        return img


def _shuffle_lists(rng, lists):
    """
    shuffles, inplace, multiple lists/arrays together so that
    corresponding items in each list are still corresponding
    after shuffle
    """
    length = len(lists[0])
    for l in lists:
        assert len(l) == length

    for i in xrange(length):
        j = rng.randint(length - 1)
        for l in lists:             # must make copy in case l is an array:
            tmp = l[i].copy()       # l[i], l[j] = l[j], l[i] will not work
            l[i] = l[j]             # because l[i] is written over
            l[j] = tmp              # before l[j] can receive its value


