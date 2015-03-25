__author__ = 'isaac'

import smtplib
import sys
from email.mime.text import MIMEText
import os
import numpy
from StringIO import StringIO
import datetime
import time
import yaml


def load_config(environ_variable, module_file, default_config):
    default_config = os.path.join(os.path.dirname(module_file), default_config)
    print default_config
    return yaml.load(file(os.getenv(environ_variable, default_config)))


def send_email(
        from_addr,
        to_addr,
        username,
        password,
        smtp,
        subject='',
        body='',):
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = from_addr
    msg['To'] = to_addr

    server = smtplib.SMTP(smtp)
    server.starttls()
    server.login(username, password)
    server.sendmail(from_addr, to_addr, msg.as_string())
    server.quit()


class Dot(object):
    def __init__(self, skip=0):
        self.number = 0
        self.skip_count = int(skip)
        self.num_skip = int(skip)

    def stop(self):
        sys.stdout.write(' done\n')
        sys.stdout.flush()

    def dot(self, string=None):
        if self.skip_count == self.num_skip:
            self.skip_count = 0
            string = "." if string is None else string
            self.number += 1

            if self.number > 40:
                sys.stdout.write("\n")
                self.number = 0

            sys.stdout.write(string)
            sys.stdout.flush()
        else:
            self.skip_count += 1


def raise_exception(x):
    raise Exception(x)


def get_sub_dirs(directory, rel=False, cache=False):
    path, subs, files = next(os.walk(directory, onerror=raise_exception))
    key = directory + str(rel)
    if cache and key in get_sub_dirs.cache:
        return get_sub_dirs.cache[key]
    if not rel:
        subs = [path + os.sep + sub for sub in subs]
    if cache:
        get_sub_dirs.cache[key] = subs
    return subs
get_sub_dirs.cache = {}


def get_files(directory, rel=False, cache=False):
    path, subs, files = next(os.walk(directory, onerror=raise_exception))
    key = directory + str(rel)
    if cache and key in get_files.cache:
        return get_files.cache[key]
    if not rel:
        files = [path + os.sep + f for f in files]
    if cache:
        get_files.cache[key] = files
    return files
get_files.cache = {}


def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def image_tile(X, img_shape, tile_shape, tile_spacing=(0, 0),
               scale_rows_to_unit_interval=True,
               output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
        be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
        values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
        being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
        (See:`Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
    ]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype='uint8')
        else:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in range(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = numpy.zeros(
                    out_shape,
                    dtype=dt
                ) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = image_tile(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = numpy.zeros(out_shape, dtype=dt)

        for tile_row in range(tile_shape[0]):
            for tile_col in range(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] = this_img * c
        return out_array


def tile_2d_images(images, canvas_shape):
    height = images.shape[1]
    width = images.shape[2]
    canvas = numpy.zeros(
        ((height+1)*canvas_shape[0], (width+1)*canvas_shape[1]),
        dtype=images.dtype)

    index = 0
    for i in range(canvas_shape[0]):
        for j in range(canvas_shape[1]):
            if index >= images.shape[0]:
                return canvas
            v_off = i*(height+1)
            h_off = j*(width+1)
            canvas[v_off:v_off + height, h_off:h_off+width] = images[index]
            index += 1

    return canvas


def default(variable, dfault):
    return dfault if variable is None else variable


def num_abbrev(num, abbrev, sep):
    import math
    if num < 1:
        return num
    num = float(num)
    millidx = max(0, min(len(abbrev) - 1,
                         int(math.floor(math.log10(abs(num))/3))))
    return '{:.0f}{}{}'.format(num / 10 ** (3 * millidx), sep, abbrev[millidx])


def human(num):
    return num_abbrev(num,
                      ['', 'Thousand', 'Million', 'Billion', 'Trillion',
                       'Quadrillion', 'Quintillion', 'TOO BIG'], ' ')


def hum(num):
    return num_abbrev(num,
                      ['', 'thou', 'mill', 'bill', 'trill',
                       'quad', 'quin', 'TOO BIG'], ' ')


def H(num):
    return num_abbrev(num,
                      ['', 't', 'M', 'B', 'T',
                       'Q', 'N', 'TOO BIG'], '')


def h(num):
    return num_abbrev(num,
                      [' ', 't', 'm', 'b', 'tr',
                       'qd', 'qn', '##'], '')


def now():
    return ((datetime.datetime.utcnow() - datetime.timedelta(hours=5))
            .strftime('%Y-%m-%d %I:%M:%S %p')) + ' (EST)'


def save_output(filename, func, *args, **kw):
    class SplitIO(StringIO):
        def __init__(self, term):
            StringIO.__init__(self)
            self.term = term

        def write(self, output):
            StringIO.write(self, output)
            self.term.write(output)

        def flush(self):
            StringIO.flush(self)
            self.term.flush()

    term_out, term_error = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = SplitIO(term_out), SplitIO(term_error)
        func(*args, **kw)
    finally:
        out, err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = term_out, term_error
        out, err = out.getvalue(), err.getvalue()
        with open(filename, 'w') as f:
            f.write(out + "\n---------------------------\n" + err)


def time_once(method):

    def timed(*args, **kw):
        if timed.first:
            ts = time.time()

        result = method(*args, **kw)

        if timed.first:
            te = time.time()

            # print '%r (%r, %r) %2.2f sec' % \
            #       (method.__name__, args, kw, te-ts)

            print('func: {}; time: {}'.format(method.__name__, te - ts))

            timed.first = False
        return result

    timed.first = True

    return timed