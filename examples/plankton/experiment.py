__author__ = 'isaac'

import os

from examples.plankton import plankton
from pydnn import nn
from pydnn import preprocess as pp
from pydnn import tools


class Experiment(object):
    """
    comments: like 63 except:
    (1) using overlapping pools
    (2) adam
    (3) LearningRateDecay

    results:

    """
    def __init__(p):
        p.name = 'e082'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 237
        p.epochs = 10000
        p.final_epochs = 40
        p.image_shape = (77, 77)

        p.rng_seed = 13579

        p.learning_rule = nn.Adam(
            b1=0.9,
            b2=0.999,
            e=1e-8,
            lmbda=1 - 1e-8,
            learning_rate=nn.LearningRateDecay(
                learning_rate=0.006,
                decay=.05))

        p.activation = nn.relu

        p.batch_normalize = True

        p.l1_reg = 0.000
        p.l2_reg = 0.00005
        p.dropout_conv = 0.35
        p.dropout_hidd = 0.55

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360

    def build_net(p, n):
        n.add_convolution(64, (7, 7), (3, 3), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(96, (5, 5), (3, 3), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(96, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_conv_pool(128, (3, 3), (3, 3), (2, 2), use_bias=False)
        n.merge_data_channel('shapes')
        n.add_batch_normalization()
        n.add_nonlinearity(p.activation)
        n.add_dropout(p.dropout_conv)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_logistic()

        """
        type | calcs | params |    neurons     |     output     | mults | adds | weights | biases | pool loss
        ------------------------------------------------------------------------------------------------------
        conv |   32m |     3t |   (64, 71, 71) |   (64, 35, 35) |   16m |  16m |      3t |    64  |
        conv |  295m |     2t |   (96, 31, 31) |   (96, 15, 15) |  148m | 148m |      2t |    96  |
        conv |   28m |   960  |   (96, 13, 13) |   (96, 13, 13) |   14m |  14m |    864  |    96  |
        conv |   27m |     1t |  (128, 11, 11) |  (128, 11, 11) |   13m |  13m |      1t |   128  |
        conv |   24m |     1t |    (128, 9, 9) |    (128, 4, 4) |   12m |  12m |      1t |   128  |
        full |   13m |     6m |           3072 |           3072 |    6m |   6m |      6m |     3t |
        full |   19m |     9m |           3072 |           3072 |    9m |   9m |      9m |     3t |
        ----------------------------------------
        total   437m      16m           463104

        7.29 mins (+/- 1 order of magnitude) per epoch estimated for ec2 g2
        120.1 mb estimated for model
        """


class Experiment097(object):
    """
    comments: like e096 except:
    (1) more splits

    remember to try turning dropout down... I'm removing a lot of connections
    from the network so that might work as a regularizer.

    (based on 82)

    results:
    The point of this experiment was to speed up training but instead
    each epoch takes twice as long.  I wonder why...

    """
    def __init__(p):
        p.name = 'e097'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 237
        p.epochs = 300
        p.final_epochs = 0
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (77, 77)

        p.rng_seed = 13579

        p.learning_rule = nn.Adam(
            b1=0.9,
            b2=0.999,
            e=1e-8,
            lmbda=1 - 1e-8,
            learning_rate=nn.LearningRateDecay(
                learning_rate=0.006,
                decay=.05))

        p.activation = nn.relu

        p.batch_normalize = True

        p.l1_reg = 0.000
        p.l2_reg = 0.00005
        p.dropout_conv = 0.35
        p.dropout_hidd = 0.55

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360

    def build_net(p, n):
        split0 = n.split_pathways(2)
        for pw0 in split0 + [n]:
            pw0.add_convolution(32, (7, 7), (3, 3), (2, 2), batch_normalize=p.batch_normalize)
            pw0.add_dropout(p.dropout_conv)
            pw0.add_convolution(48, (5, 5), (3, 3), (2, 2), batch_normalize=p.batch_normalize)
            pw0.add_dropout(p.dropout_conv)
            split5 = pw0.split_pathways(2)
            for pw5 in split5 + [pw0]:
                pw5.add_convolution(48, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
                pw5.add_dropout(p.dropout_conv)
                split7 = pw5.split_pathways(2)
                for pw7 in split7 + [pw5]:
                    pw7.add_convolution(32, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
                    pw7.add_dropout(p.dropout_conv)
                    pw7.add_convolution(32, (3, 3), (3, 3), (2, 2), batch_normalize=p.batch_normalize)
                    pw7.add_dropout(p.dropout_conv)
                    pw7.merge_data_channel('shapes')
                pw5.merge_pathways(split7)
                pw5.add_hidden(1536, batch_normalize=p.batch_normalize)
                pw5.add_dropout(p.dropout_hidd)
            pw0.merge_pathways(split5)
        n.merge_pathways(split0)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_logistic()


class Experiment096(object):
    """
    comments: like 63 except:
    (1) using overlapping pools
    (2) adam w/ decay
    (3) split pathways

    remember to try turning dropout down... I'm removing a lot of connections
    from the network so that might work as a regularizer.

    (based on 82)

    results:
    looks like it is generalizing terribly and training worse.
    I also tried a variant with just a two way split and it was
    generalizing a substantially better (though still much worse than
    without the splits).  I halved the number of filters in each layer so
    I would have expected a two way split to generalize more similarly to
    a splitless architecture (because with a two way split with each split
    having half the number of filters there is about the same number of filters...
    so there aren't a lot more parameters with which to overfit).  The
    extremely poor generalization is a bit of a mystery to me.   Could it be
    a bug?

    """
    def __init__(p):
        p.name = 'e096'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 237
        p.epochs = 300
        p.final_epochs = 0
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (77, 77)

        p.rng_seed = 13579

        p.learning_rule = nn.Adam(
            b1=0.9,
            b2=0.999,
            e=1e-8,
            lmbda=1 - 1e-8,
            learning_rate=nn.LearningRateDecay(
                learning_rate=0.006,
                decay=.05))

        p.activation = nn.relu

        p.batch_normalize = True

        p.l1_reg = 0.000
        p.l2_reg = 0.00005
        p.dropout_conv = 0.35
        p.dropout_hidd = 0.55

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360

    def build_net(p, n):
        n.add_convolution(64, (7, 7), (3, 3), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(96, (5, 5), (3, 3), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        split1 = n.split_pathways(2)
        for pw1 in split1 + [n]:
            pw1.add_convolution(48, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
            pw1.add_dropout(p.dropout_conv)
            pw1.add_convolution(64, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
            pw1.add_dropout(p.dropout_conv)
            pw1.add_convolution(64, (3, 3), (3, 3), (2, 2), batch_normalize=p.batch_normalize)
            pw1.add_dropout(p.dropout_conv)
            pw1.merge_data_channel('shapes')
            pw1.add_hidden(1536, batch_normalize=p.batch_normalize)
            pw1.add_dropout(p.dropout_hidd)
        n.merge_pathways(split1)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_logistic()

#############################################################
# Experiments Above This Line Added After Competition Ended #
#############################################################

class Experiment095(object):
    """
    comments:

    like 94 but:
    (1) original images feeds into fully connected layers
    (2) shapes come after dropout

    like 68 except:
    (0) original images feeds into fully connected layers
    (1) shapes (not geometry)
    (2) using overlapping pools
    (1) adam
    (2) .05 learning rate decay
    (3) 192 filters in final conv
    (4) .006 initial learning rate
    (5) lower dropout

    results:


    """
    def __init__(p):
        p.name = 'e095'
        p.num_images = None
        p.train_pct = 90
        p.valid_pct = 5
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 237
        p.epochs = 200
        p.final_epochs = 0
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (77, 77)

        p.rng_seed = 13579

        p.learning_rule = nn.Adam(
            b1=0.9,
            b2=0.999,
            e=1e-8,
            lmbda=1 - 1e-8,
            learning_rate_annealer=nn.LearningRateDecay(
                learning_rate=0.006,
                decay=.1))

        p.activation = nn.relu

        p.batch_normalize = True

        p.l1_reg = 0.000
        p.l2_reg = 0.00005
        p.dropout_conv = 0.4
        p.dropout_hidd = 0.5

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360

    def build_net(p, n):
        n.add_convolution(72, (7, 7), (3, 3), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(0.5)
        n.add_convolution(128, (5, 5), (3, 3), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(0.4)
        n.add_convolution(128, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(0.3)
        n.add_convolution(192, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(0.2)
        n.add_conv_pool(128, (3, 3), (3, 3), (2, 2), use_bias=False)
        n.add_batch_normalization()
        n.add_nonlinearity(p.activation)
        n.add_dropout(p.dropout_conv)
        n.merge_data_channel('shapes')
        n.merge_data_channel('images')
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_logistic()


        """
        type | calcs | params |    neurons     |     output     | mults | adds | weights | biases | pool loss
        ------------------------------------------------------------------------------------------------------
        conv |   36m |     4t |   (72, 71, 71) |   (72, 35, 35) |   18m |  18m |      4t |    72  |
        conv |  443m |     3t |  (128, 31, 31) |  (128, 15, 15) |  221m | 222m |      3t |   128  |
        conv |   50m |     1t |  (128, 13, 13) |  (128, 13, 13) |   25m |  25m |      1t |   128  |
        conv |   54m |     2t |  (192, 11, 11) |  (192, 11, 11) |   27m |  27m |      2t |   192  |
        conv |   36m |     1t |    (128, 9, 9) |    (128, 4, 4) |   18m |  18m |      1t |   128  |
        full |   13m |     6m |           3072 |           3072 |    6m |   6m |      6m |     3t |
        full |   19m |     9m |           3072 |           3072 |    9m |   9m |      9m |     3t |
        full |   19m |     9m |           3072 |           3072 |    9m |   9m |      9m |     3t |
        ----------------------------------------
        total   668m      25m           550408

        11.14 mins (+/- 1 order of magnitude) per epoch estimated for ec2 g2
        192.2 mb estimated for model
        """

class Experiment094(object):
    """
    comments:
    like 93 but:
    (1) .05 lower decay

    like 90 but:
    (2) .05 learning rate decay
    (5) lower dropout

    like 68 except:
    (1) shapes (not geometry)
    (2) using overlapping pools
    (1) adam
    (2) .05 learning rate decay
    (3) 192 filters in final conv
    (4) .006 initial learning rate
    (5) lower dropout

    results:
    Looks like it stopped while it was still improving rapidly so I'm running a
    few more epochs:
        >>> from examples.plankton import experiment plankton        >>> from pydnn import nn        >>> import examples.plankton.plankton
        >>> images, classes, file_indices, labels = plankton.build_training_set()
        >>> data = (images, classes, file_indices)
        >>> e = experiment.Experiment()
        >>> import preprocessors as pp
        >>> data = pp.split_training_data(data, e.batch_size, e.train_pct, e.valid_pct, e.test_pct)
        >>> net = nn.load('e094_final_net.pkl')
        >>> net.preprocessor.set_data(data)
        >>> updater = nn.Adam(
        ...     b1=0.9,
        ...     b2=0.999,
        ...     e=1e-8,
        ...     lmbda=1 - 1e-8,
        ...     learning_rate_annealer=nn.LearningRateDecay(
        ...         learning_rate=0.0006,
        ...         decay=.05))
        >>> net.train(updater, 500, 0, e.l1_reg, e.l2_reg)
    """
    def __init__(p):
        p.name = 'e094'
        p.num_images = None
        p.train_pct = 90
        p.valid_pct = 5
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 237
        p.epochs = 200
        p.final_epochs = 0
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (77, 77)

        p.rng_seed = 13579

        p.learning_rule = nn.Adam(
            b1=0.9,
            b2=0.999,
            e=1e-8,
            lmbda=1 - 1e-8,
            learning_rate_annealer=nn.LearningRateDecay(
                learning_rate=0.006,
                decay=.05))

        p.activation = nn.relu

        p.batch_normalize = True

        p.l1_reg = 0.000
        p.l2_reg = 0.00005
        p.dropout_conv = 0.4
        p.dropout_hidd = 0.5

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360

    def build_net(p, n):
        n.add_convolution(72, (7, 7), (3, 3), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(0.5)
        n.add_convolution(128, (5, 5), (3, 3), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(0.4)
        n.add_convolution(128, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(0.3)
        n.add_convolution(192, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(0.2)
        n.add_conv_pool(192, (3, 3), (3, 3), (2, 2), use_bias=False)
        n.merge_data_channel('shapes')
        n.add_batch_normalization()
        n.add_nonlinearity(p.activation)
        n.add_dropout(p.dropout_conv)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_logistic()


        """
        type | calcs | params |    neurons     |     output     | mults | adds | weights | biases | pool loss
        ------------------------------------------------------------------------------------------------------
        conv |   36m |     4t |   (72, 71, 71) |   (72, 35, 35) |   18m |  18m |      4t |    72  |
        conv |  443m |     3t |  (128, 31, 31) |  (128, 15, 15) |  221m | 222m |      3t |   128  |
        conv |   50m |     1t |  (128, 13, 13) |  (128, 13, 13) |   25m |  25m |      1t |   128  |
        conv |   54m |     2t |  (192, 11, 11) |  (192, 11, 11) |   27m |  27m |      2t |   192  |
        conv |   36m |     1t |    (128, 9, 9) |    (128, 4, 4) |   18m |  18m |      1t |   128  |
        full |   13m |     6m |           3072 |           3072 |    6m |   6m |      6m |     3t |
        full |   19m |     9m |           3072 |           3072 |    9m |   9m |      9m |     3t |
        full |   19m |     9m |           3072 |           3072 |    9m |   9m |      9m |     3t |
        ----------------------------------------
        total   668m      25m           550408

        11.14 mins (+/- 1 order of magnitude) per epoch estimated for ec2 g2
        192.2 mb estimated for model
        """


class Experiment093(object):
    """
    comments:
    like 90 but:
    (2) .5 learning rate decay
    (5) lower dropout

    like 68 except:
    (1) shapes (not geometry)
    (2) using overlapping pools
    (1) adam
    (2) .5 learning rate decay
    (3) 192 filters in final conv
    (4) .006 initial learning rate
    (5) lower dropout

    results:
    I think this did pretty well... but I seem to have accidentally killed the
    job without downloading the results... so I don't know for sure.  Unintiutively
    it might have been doing better than 94.  I think it ended in the 0.7s, but
    94 could still get there...


    """
    def __init__(p):
        p.name = 'e093'
        p.num_images = None
        p.train_pct = 90
        p.valid_pct = 5
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 237
        p.epochs = 200
        p.final_epochs = 0
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (77, 77)

        p.rng_seed = 13579

        p.learning_rule = nn.Adam(
            b1=0.9,
            b2=0.999,
            e=1e-8,
            lmbda=1 - 1e-8,
            learning_rate_annealer=nn.LearningRateDecay(
                learning_rate=0.006,
                decay=.5))

        p.activation = nn.relu

        p.batch_normalize = True

        p.l1_reg = 0.000
        p.l2_reg = 0.00005
        p.dropout_conv = 0.4
        p.dropout_hidd = 0.5

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360

    def build_net(p, n):
        n.add_convolution(72, (7, 7), (3, 3), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(0.5)
        n.add_convolution(128, (5, 5), (3, 3), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(0.4)
        n.add_convolution(128, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(0.3)
        n.add_convolution(192, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(0.2)
        n.add_conv_pool(192, (3, 3), (3, 3), (2, 2), use_bias=False)
        n.merge_data_channel('shapes')
        n.add_batch_normalization()
        n.add_nonlinearity(p.activation)
        n.add_dropout(p.dropout_conv)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_logistic()


        """
        type | calcs | params |    neurons     |     output     | mults | adds | weights | biases | pool loss
        ------------------------------------------------------------------------------------------------------
        conv |   36m |     4t |   (72, 71, 71) |   (72, 35, 35) |   18m |  18m |      4t |    72  |
        conv |  443m |     3t |  (128, 31, 31) |  (128, 15, 15) |  221m | 222m |      3t |   128  |
        conv |   50m |     1t |  (128, 13, 13) |  (128, 13, 13) |   25m |  25m |      1t |   128  |
        conv |   54m |     2t |  (192, 11, 11) |  (192, 11, 11) |   27m |  27m |      2t |   192  |
        conv |   36m |     1t |    (128, 9, 9) |    (128, 4, 4) |   18m |  18m |      1t |   128  |
        full |   13m |     6m |           3072 |           3072 |    6m |   6m |      6m |     3t |
        full |   19m |     9m |           3072 |           3072 |    9m |   9m |      9m |     3t |
        full |   19m |     9m |           3072 |           3072 |    9m |   9m |      9m |     3t |
        ----------------------------------------
        total   668m      25m           550408

        11.14 mins (+/- 1 order of magnitude) per epoch estimated for ec2 g2
        192.2 mb estimated for model
        """


class Experiment092(object):
    """
    comments: like 63 except:
    (0) nn.Canonicalizer
    (1) using overlapping pools
    (2) adam
    (3) .06 decay, .002 learning rate

    results:

    """
    def __init__(p):
        p.name = 'e092'
        p.num_images = None
        p.train_pct = 90
        p.valid_pct = 5
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 237
        p.epochs = 10000
        p.final_epochs = 40
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (77, 77)

        p.rng_seed = 13579

        p.learning_rule = nn.Adam(
            b1=0.9,
            b2=0.999,
            e=1e-8,
            lmbda=1 - 1e-8,
            learning_rate_annealer=nn.LearningRateDecay(
                learning_rate=0.002,
                decay=.06))

        p.activation = nn.relu

        p.batch_normalize = True

        p.l1_reg = 0.000
        p.l2_reg = 0.00005
        p.dropout_conv = 0.35
        p.dropout_hidd = 0.55

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Canonicalizer

    def build_net(p, n):
        n.add_convolution(64, (7, 7), (3, 3), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(96, (5, 5), (3, 3), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(96, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_conv_pool(128, (3, 3), (3, 3), (2, 2), use_bias=False)
        n.merge_data_channel('geometry')
        n.add_batch_normalization()
        n.add_nonlinearity(p.activation)
        n.add_dropout(p.dropout_conv)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_logistic()

        """
        type | calcs | params |    neurons     |     output     | mults | adds | weights | biases | pool loss
        ------------------------------------------------------------------------------------------------------
        conv |   32m |     3t |   (64, 71, 71) |   (64, 35, 35) |   16m |  16m |      3t |    64  |
        conv |  295m |     2t |   (96, 31, 31) |   (96, 15, 15) |  148m | 148m |      2t |    96  |
        conv |   28m |   960  |   (96, 13, 13) |   (96, 13, 13) |   14m |  14m |    864  |    96  |
        conv |   27m |     1t |  (128, 11, 11) |  (128, 11, 11) |   13m |  13m |      1t |   128  |
        conv |   24m |     1t |    (128, 9, 9) |    (128, 4, 4) |   12m |  12m |      1t |   128  |
        full |   13m |     6m |           3072 |           3072 |    6m |   6m |      6m |     3t |
        full |   19m |     9m |           3072 |           3072 |    9m |   9m |      9m |     3t |
        ----------------------------------------
        total   437m      16m           463104

        7.29 mins (+/- 1 order of magnitude) per epoch estimated for ec2 g2
        120.1 mb estimated for model
        """


class Experiment091(object):
    """
    comments:
    like 86 but:
    (1) using pp.Canonicalizer

    like 68 except:
    (1) using pp.Canonicalizer
    (2) using overlapping pools
    (1) adam
    (2) .06 learning rate decay
    (3) 128 filters in final conv

    results:


    """
    def __init__(p):
        p.name = 'e091'
        p.num_images = None
        p.train_pct = 90
        p.valid_pct = 5
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 237
        p.epochs = 200
        p.final_epochs = 0
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (77, 77)

        p.rng_seed = 13579

        p.learning_rule = nn.Adam(
            b1=0.9,
            b2=0.999,
            e=1e-8,
            lmbda=1 - 1e-8,
            learning_rate_annealer=nn.LearningRateDecay(
                learning_rate=0.002,
                decay=.06))

        p.activation = nn.relu

        p.batch_normalize = True

        p.l1_reg = 0.000
        p.l2_reg = 0.00005
        p.dropout_conv = 0.45
        p.dropout_hidd = 0.6

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Canonicalizer

    def build_net(p, n):
        n.add_convolution(72, (7, 7), (3, 3), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (5, 5), (3, 3), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(192, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_conv_pool(128, (3, 3), (3, 3), (2, 2), use_bias=False)
        n.merge_data_channel('geometry')
        n.add_batch_normalization()
        n.add_nonlinearity(p.activation)
        n.add_dropout(p.dropout_conv)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_logistic()


        """
        type | calcs | params |    neurons     |     output     | mults | adds | weights | biases | pool loss
        ------------------------------------------------------------------------------------------------------
        conv |   36m |     4t |   (72, 71, 71) |   (72, 35, 35) |   18m |  18m |      4t |    72  |
        conv |  443m |     3t |  (128, 31, 31) |  (128, 15, 15) |  221m | 222m |      3t |   128  |
        conv |   50m |     1t |  (128, 13, 13) |  (128, 13, 13) |   25m |  25m |      1t |   128  |
        conv |   54m |     2t |  (192, 11, 11) |  (192, 11, 11) |   27m |  27m |      2t |   192  |
        conv |   36m |     1t |    (128, 9, 9) |    (128, 4, 4) |   18m |  18m |      1t |   128  |
        full |   13m |     6m |           3072 |           3072 |    6m |   6m |      6m |     3t |
        full |   19m |     9m |           3072 |           3072 |    9m |   9m |      9m |     3t |
        full |   19m |     9m |           3072 |           3072 |    9m |   9m |      9m |     3t |
        ----------------------------------------
        total   668m      25m           550408

        11.14 mins (+/- 1 order of magnitude) per epoch estimated for ec2 g2
        192.2 mb estimated for model
        """


class Experiment090(object):
    """
    comments:

    like 68 except:
    (1) shapes (not geometry)
    (2) using overlapping pools
    (1) adam
    (2) .1 learning rate decay
    (3) 192 filters in final conv
    (4) .006 initial learning rate
    (5) lower dropout

    results:


    """
    def __init__(p):
        p.name = 'e090'
        p.num_images = None
        p.train_pct = 90
        p.valid_pct = 5
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 237
        p.epochs = 200
        p.final_epochs = 0
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (77, 77)

        p.rng_seed = 13579

        p.learning_rule = nn.Adam(
            b1=0.9,
            b2=0.999,
            e=1e-8,
            lmbda=1 - 1e-8,
            learning_rate_annealer=nn.LearningRateDecay(
                learning_rate=0.006,
                decay=.1))

        p.activation = nn.relu

        p.batch_normalize = True

        p.l1_reg = 0.000
        p.l2_reg = 0.00005
        p.dropout_conv = 0.4
        p.dropout_hidd = 0.55

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360

    def build_net(p, n):
        n.add_convolution(72, (7, 7), (3, 3), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (5, 5), (3, 3), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(192, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_conv_pool(192, (3, 3), (3, 3), (2, 2), use_bias=False)
        n.merge_data_channel('shapes')
        n.add_batch_normalization()
        n.add_nonlinearity(p.activation)
        n.add_dropout(p.dropout_conv)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_logistic()


        """
        type | calcs | params |    neurons     |     output     | mults | adds | weights | biases | pool loss
        ------------------------------------------------------------------------------------------------------
        conv |   36m |     4t |   (72, 71, 71) |   (72, 35, 35) |   18m |  18m |      4t |    72  |
        conv |  443m |     3t |  (128, 31, 31) |  (128, 15, 15) |  221m | 222m |      3t |   128  |
        conv |   50m |     1t |  (128, 13, 13) |  (128, 13, 13) |   25m |  25m |      1t |   128  |
        conv |   54m |     2t |  (192, 11, 11) |  (192, 11, 11) |   27m |  27m |      2t |   192  |
        conv |   36m |     1t |    (128, 9, 9) |    (128, 4, 4) |   18m |  18m |      1t |   128  |
        full |   13m |     6m |           3072 |           3072 |    6m |   6m |      6m |     3t |
        full |   19m |     9m |           3072 |           3072 |    9m |   9m |      9m |     3t |
        full |   19m |     9m |           3072 |           3072 |    9m |   9m |      9m |     3t |
        ----------------------------------------
        total   668m      25m           550408

        11.14 mins (+/- 1 order of magnitude) per epoch estimated for ec2 g2
        192.2 mb estimated for model
        """



class Experiment089(object):
    """
    comments: like 63 except:
    (0) shapes (not geometry)
    (1) using overlapping pools
    (2) adam
    (3) .06 decay, .002 learning rate

    results:

    """
    def __init__(p):
        p.name = 'e089'
        p.num_images = None
        p.train_pct = 90
        p.valid_pct = 5
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 237
        p.epochs = 10000
        p.final_epochs = 40
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (77, 77)

        p.rng_seed = 13579

        p.learning_rule = nn.Adam(
            b1=0.9,
            b2=0.999,
            e=1e-8,
            lmbda=1 - 1e-8,
            learning_rate_annealer=nn.LearningRateDecay(
                learning_rate=0.002,
                decay=.06))

        p.activation = nn.relu

        p.batch_normalize = True

        p.l1_reg = 0.000
        p.l2_reg = 0.00005
        p.dropout_conv = 0.35
        p.dropout_hidd = 0.55

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360PlusGeometry

    def build_net(p, n):
        n.add_convolution(64, (7, 7), (3, 3), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(96, (5, 5), (3, 3), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(96, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_conv_pool(128, (3, 3), (3, 3), (2, 2), use_bias=False)
        n.merge_data_channel('geometry')
        n.add_batch_normalization()
        n.add_nonlinearity(p.activation)
        n.add_dropout(p.dropout_conv)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_logistic()

        """
        type | calcs | params |    neurons     |     output     | mults | adds | weights | biases | pool loss
        ------------------------------------------------------------------------------------------------------
        conv |   32m |     3t |   (64, 71, 71) |   (64, 35, 35) |   16m |  16m |      3t |    64  |
        conv |  295m |     2t |   (96, 31, 31) |   (96, 15, 15) |  148m | 148m |      2t |    96  |
        conv |   28m |   960  |   (96, 13, 13) |   (96, 13, 13) |   14m |  14m |    864  |    96  |
        conv |   27m |     1t |  (128, 11, 11) |  (128, 11, 11) |   13m |  13m |      1t |   128  |
        conv |   24m |     1t |    (128, 9, 9) |    (128, 4, 4) |   12m |  12m |      1t |   128  |
        full |   13m |     6m |           3072 |           3072 |    6m |   6m |      6m |     3t |
        full |   19m |     9m |           3072 |           3072 |    9m |   9m |      9m |     3t |
        ----------------------------------------
        total   437m      16m           463104

        7.29 mins (+/- 1 order of magnitude) per epoch estimated for ec2 g2
        120.1 mb estimated for model
        """


class Experiment088(object):
    """
    comments:

    like 68 except:
    (1) shapes (not geometry)
    (2) using overlapping pools
    (1) adam
    (2) .1 learning rate decay
    (3) 192 filters in final conv

    results:


    """
    def __init__(p):
        p.name = 'e088'
        p.num_images = None
        p.train_pct = 90
        p.valid_pct = 5
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 237
        p.epochs = 200
        p.final_epochs = 0
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (77, 77)

        p.rng_seed = 13579

        p.learning_rule = nn.Adam(
            b1=0.9,
            b2=0.999,
            e=1e-8,
            lmbda=1 - 1e-8,
            learning_rate_annealer=nn.LearningRateDecay(
                learning_rate=0.004,
                decay=.1))

        p.activation = nn.relu

        p.batch_normalize = True

        p.l1_reg = 0.000
        p.l2_reg = 0.00005
        p.dropout_conv = 0.45
        p.dropout_hidd = 0.6

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360

    def build_net(p, n):
        n.add_convolution(72, (7, 7), (3, 3), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (5, 5), (3, 3), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(192, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_conv_pool(192, (3, 3), (3, 3), (2, 2), use_bias=False)
        n.merge_data_channel('shapes')
        n.add_batch_normalization()
        n.add_nonlinearity(p.activation)
        n.add_dropout(p.dropout_conv)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_logistic()


        """
        type | calcs | params |    neurons     |     output     | mults | adds | weights | biases | pool loss
        ------------------------------------------------------------------------------------------------------
        conv |   36m |     4t |   (72, 71, 71) |   (72, 35, 35) |   18m |  18m |      4t |    72  |
        conv |  443m |     3t |  (128, 31, 31) |  (128, 15, 15) |  221m | 222m |      3t |   128  |
        conv |   50m |     1t |  (128, 13, 13) |  (128, 13, 13) |   25m |  25m |      1t |   128  |
        conv |   54m |     2t |  (192, 11, 11) |  (192, 11, 11) |   27m |  27m |      2t |   192  |
        conv |   36m |     1t |    (128, 9, 9) |    (128, 4, 4) |   18m |  18m |      1t |   128  |
        full |   13m |     6m |           3072 |           3072 |    6m |   6m |      6m |     3t |
        full |   19m |     9m |           3072 |           3072 |    9m |   9m |      9m |     3t |
        full |   19m |     9m |           3072 |           3072 |    9m |   9m |      9m |     3t |
        ----------------------------------------
        total   668m      25m           550408

        11.14 mins (+/- 1 order of magnitude) per epoch estimated for ec2 g2
        192.2 mb estimated for model
        """



class Experiment087(object):
    """
    comments:

    like 68 except:
    (1) shapes (not geometry)
    (2) using overlapping pools
    (1) adam
    (2) .06 learning rate decay
    (3) 192 filters in final conv

    results:


    """
    def __init__(p):
        p.name = 'e087'
        p.num_images = None
        p.train_pct = 90
        p.valid_pct = 5
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 237
        p.epochs = 200
        p.final_epochs = 0
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (77, 77)

        p.rng_seed = 13579

        p.learning_rule = nn.Adam(
            b1=0.9,
            b2=0.999,
            e=1e-8,
            lmbda=1 - 1e-8,
            learning_rate_annealer=nn.LearningRateDecay(
                learning_rate=0.002,
                decay=.06))

        p.activation = nn.relu

        p.batch_normalize = True

        p.l1_reg = 0.000
        p.l2_reg = 0.00005
        p.dropout_conv = 0.45
        p.dropout_hidd = 0.6

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360

    def build_net(p, n):
        n.add_convolution(72, (7, 7), (3, 3), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (5, 5), (3, 3), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(192, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_conv_pool(192, (3, 3), (3, 3), (2, 2), use_bias=False)
        n.merge_data_channel('shapes')
        n.add_batch_normalization()
        n.add_nonlinearity(p.activation)
        n.add_dropout(p.dropout_conv)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_logistic()


        """
        type | calcs | params |    neurons     |     output     | mults | adds | weights | biases | pool loss
        ------------------------------------------------------------------------------------------------------
        conv |   36m |     4t |   (72, 71, 71) |   (72, 35, 35) |   18m |  18m |      4t |    72  |
        conv |  443m |     3t |  (128, 31, 31) |  (128, 15, 15) |  221m | 222m |      3t |   128  |
        conv |   50m |     1t |  (128, 13, 13) |  (128, 13, 13) |   25m |  25m |      1t |   128  |
        conv |   54m |     2t |  (192, 11, 11) |  (192, 11, 11) |   27m |  27m |      2t |   192  |
        conv |   36m |     1t |    (128, 9, 9) |    (128, 4, 4) |   18m |  18m |      1t |   128  |
        full |   13m |     6m |           3072 |           3072 |    6m |   6m |      6m |     3t |
        full |   19m |     9m |           3072 |           3072 |    9m |   9m |      9m |     3t |
        full |   19m |     9m |           3072 |           3072 |    9m |   9m |      9m |     3t |
        ----------------------------------------
        total   668m      25m           550408

        11.14 mins (+/- 1 order of magnitude) per epoch estimated for ec2 g2
        192.2 mb estimated for model
        """


class Experiment086(object):
    """
    comments:

    like 68 except:
    (1) using pp.Rotator360PlusGeometry
    (2) using overlapping pools
    (1) adam
    (2) .06 learning rate decay
    (3) 128 filters in final conv

    results:


    """
    def __init__(p):
        p.name = 'e086'
        p.num_images = None
        p.train_pct = 90
        p.valid_pct = 5
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 237
        p.epochs = 200
        p.final_epochs = 0
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (77, 77)

        p.rng_seed = 13579

        p.learning_rule = nn.Adam(
            b1=0.9,
            b2=0.999,
            e=1e-8,
            lmbda=1 - 1e-8,
            learning_rate_annealer=nn.LearningRateDecay(
                learning_rate=0.002,
                decay=.06))

        p.activation = nn.relu

        p.batch_normalize = True

        p.l1_reg = 0.000
        p.l2_reg = 0.00005
        p.dropout_conv = 0.45
        p.dropout_hidd = 0.6

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360PlusGeometry

    def build_net(p, n):
        n.add_convolution(72, (7, 7), (3, 3), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (5, 5), (3, 3), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(192, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_conv_pool(128, (3, 3), (3, 3), (2, 2), use_bias=False)
        n.merge_data_channel('geometry')
        n.add_batch_normalization()
        n.add_nonlinearity(p.activation)
        n.add_dropout(p.dropout_conv)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_logistic()


        """
        type | calcs | params |    neurons     |     output     | mults | adds | weights | biases | pool loss
        ------------------------------------------------------------------------------------------------------
        conv |   36m |     4t |   (72, 71, 71) |   (72, 35, 35) |   18m |  18m |      4t |    72  |
        conv |  443m |     3t |  (128, 31, 31) |  (128, 15, 15) |  221m | 222m |      3t |   128  |
        conv |   50m |     1t |  (128, 13, 13) |  (128, 13, 13) |   25m |  25m |      1t |   128  |
        conv |   54m |     2t |  (192, 11, 11) |  (192, 11, 11) |   27m |  27m |      2t |   192  |
        conv |   36m |     1t |    (128, 9, 9) |    (128, 4, 4) |   18m |  18m |      1t |   128  |
        full |   13m |     6m |           3072 |           3072 |    6m |   6m |      6m |     3t |
        full |   19m |     9m |           3072 |           3072 |    9m |   9m |      9m |     3t |
        full |   19m |     9m |           3072 |           3072 |    9m |   9m |      9m |     3t |
        ----------------------------------------
        total   668m      25m           550408

        11.14 mins (+/- 1 order of magnitude) per epoch estimated for ec2 g2
        192.2 mb estimated for model
        """



class Experiment085(object):
    """
    comments:
    like e080 but:
    (1) train on whole dataset

    like 68 except:
    (1) using pp.Rotator360PlusGeometry
    (2) using overlapping pools
    (1) adam
    (2) learning rate decay

    results:


    """
    def __init__(p):
        p.name = 'e085'
        p.num_images = None
        p.train_pct = 100
        p.valid_pct = 0
        p.test_pct = 0
        p.num_submission_images = None

        p.batch_size = 237
        p.epochs = 200
        p.final_epochs = 0
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (77, 77)

        p.rng_seed = 13579

        p.learning_rule = nn.Adam(
            b1=0.9,
            b2=0.999,
            e=1e-8,
            lmbda=1 - 1e-8,
            learning_rate_annealer=nn.LearningRateDecay(
                learning_rate=0.001,
                decay=.03))

        p.activation = nn.relu

        p.batch_normalize = True

        p.l1_reg = 0.000
        p.l2_reg = 0.00005
        p.dropout_conv = 0.45
        p.dropout_hidd = 0.6

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360PlusGeometry

    def build_net(p, n):
        n.add_convolution(72, (7, 7), (3, 3), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (5, 5), (3, 3), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(192, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_conv_pool(128, (3, 3), (3, 3), (2, 2), use_bias=False)
        n.merge_data_channel('geometry')
        n.add_batch_normalization()
        n.add_nonlinearity(p.activation)
        n.add_dropout(p.dropout_conv)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_logistic()


        """
        type | calcs | params |    neurons     |     output     | mults | adds | weights | biases | pool loss
        ------------------------------------------------------------------------------------------------------
        conv |   36m |     4t |   (72, 71, 71) |   (72, 35, 35) |   18m |  18m |      4t |    72  |
        conv |  443m |     3t |  (128, 31, 31) |  (128, 15, 15) |  221m | 222m |      3t |   128  |
        conv |   50m |     1t |  (128, 13, 13) |  (128, 13, 13) |   25m |  25m |      1t |   128  |
        conv |   54m |     2t |  (192, 11, 11) |  (192, 11, 11) |   27m |  27m |      2t |   192  |
        conv |   36m |     1t |    (128, 9, 9) |    (128, 4, 4) |   18m |  18m |      1t |   128  |
        full |   13m |     6m |           3072 |           3072 |    6m |   6m |      6m |     3t |
        full |   19m |     9m |           3072 |           3072 |    9m |   9m |      9m |     3t |
        full |   19m |     9m |           3072 |           3072 |    9m |   9m |      9m |     3t |
        ----------------------------------------
        total   668m      25m           550408

        11.14 mins (+/- 1 order of magnitude) per epoch estimated for ec2 g2
        192.2 mb estimated for model
        """


class Experiment084(object):
    """
    comments:
    like e074 but:
    (1) TRAIN ON WHOLE DATASET

    like 68 but:
    (1) using pp.Rotator360PlusGeometry
    (2) TRAIN ON WHOLE DATASET

    results:

    """
    def __init__(p):
        p.name = 'e084'
        p.num_images = None
        p.train_pct = 100
        p.valid_pct = 0
        p.test_pct = 0
        p.num_submission_images = None

        p.batch_size = 237
        p.epochs = 381
        p.final_epochs = 0
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (78, 78)

        p.rng_seed = 13579

        p.learning_rule = nn.Momentum(
            initial_momentum=0.5,
            max_momentum=0.90,
            learning_rate_annealer=nn.LearningRateSchedule(
                schedule=((0, .4),
                          (115, .04),
                          (311, .004))))

        p.activation = nn.relu

        p.batch_normalize = True

        p.l1_reg = 0.000
        p.l2_reg = 0.00005
        p.dropout_conv = 0.45
        p.dropout_hidd = 0.6

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360PlusGeometry

    def build_net(p, n):
        n.add_convolution(72, (7, 7), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (5, 5), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(192, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_conv_pool(128, (3, 3), (2, 2), use_bias=False)
        n.merge_data_channel('geometry')
        n.add_batch_normalization()
        n.add_nonlinearity(p.activation)
        n.add_dropout(p.dropout_conv)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_logistic()


        """
        type | calcs | params |    neurons     |     output     | mults | adds | weights | biases | pool loss
        ------------------------------------------------------------------------------------------------------
        conv |   37m |     4t |   (72, 72, 72) |   (72, 36, 36) |   18m |  19m |      4t |    72  |
        conv |  472m |     3t |  (128, 32, 32) |  (128, 16, 16) |  236m | 236m |      3t |   128  |
        conv |   58m |     1t |  (128, 14, 14) |  (128, 14, 14) |   29m |  29m |      1t |   128  |
        conv |   64m |     2t |  (192, 12, 12) |  (192, 12, 12) |   32m |  32m |      2t |   192  |
        conv |   44m |     1t |  (128, 10, 10) |    (128, 5, 5) |   22m |  22m |      1t |   128  |
        full |   20m |    10m |           3072 |           3072 |   10m |  10m |     10m |     3t |
        full |   19m |     9m |           3072 |           3072 |    9m |   9m |      9m |     3t |
        full |   19m |     9m |           3072 |           3072 |    9m |   9m |      9m |     3t |
        ----------------------------------------
        total   732m      29m           579072

        12.20 mins (+/- 1 order of magnitude) per epoch estimated for ec2 g2
        219.2 mb estimated for model
        """


class Experiment083(object):
    """
    comments:
    like 82 but:
    (1) TRAIN ON WHOLE DATASET
    (2) learning rate schedule

    like 63 but:
    (0) using pp.Rotator360PlusGeometry
    (1) using overlapping pools
    (2) adam
    (3) learning rate schedule
    (4) TRAIN ON WHOLE DATASET
    (5) learning rate schedule

    results:

    """
    def __init__(p):
        p.name = 'e083'
        p.num_images = None
        p.train_pct = 100
        p.valid_pct = 0
        p.test_pct = 0
        p.num_submission_images = None

        p.batch_size = 237
        p.epochs = 375
        p.final_epochs = 0
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (77, 77)

        p.rng_seed = 13579

        p.learning_rule = nn.Adam(
            b1=0.9,
            b2=0.999,
            e=1e-8,
            lmbda=1 - 1e-8,
            learning_rate_annealer=nn.LearningRateSchedule(
                schedule=((0, .001),
                          (296, .0001),
                          (350, .00001))))

        p.activation = nn.relu

        p.batch_normalize = True

        p.l1_reg = 0.000
        p.l2_reg = 0.00005
        p.dropout_conv = 0.35
        p.dropout_hidd = 0.55

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360PlusGeometry

    def build_net(p, n):
        n.add_convolution(64, (7, 7), (3, 3), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(96, (5, 5), (3, 3), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(96, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_conv_pool(128, (3, 3), (3, 3), (2, 2), use_bias=False)
        n.merge_data_channel('geometry')
        n.add_batch_normalization()
        n.add_nonlinearity(p.activation)
        n.add_dropout(p.dropout_conv)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_logistic()

        """
        type | calcs | params |    neurons     |     output     | mults | adds | weights | biases | pool loss
        ------------------------------------------------------------------------------------------------------
        conv |   32m |     3t |   (64, 71, 71) |   (64, 35, 35) |   16m |  16m |      3t |    64  |
        conv |  295m |     2t |   (96, 31, 31) |   (96, 15, 15) |  148m | 148m |      2t |    96  |
        conv |   28m |   960  |   (96, 13, 13) |   (96, 13, 13) |   14m |  14m |    864  |    96  |
        conv |   27m |     1t |  (128, 11, 11) |  (128, 11, 11) |   13m |  13m |      1t |   128  |
        conv |   24m |     1t |    (128, 9, 9) |    (128, 4, 4) |   12m |  12m |      1t |   128  |
        full |   13m |     6m |           3072 |           3072 |    6m |   6m |      6m |     3t |
        full |   19m |     9m |           3072 |           3072 |    9m |   9m |      9m |     3t |
        ----------------------------------------
        total   437m      16m           463104

        7.29 mins (+/- 1 order of magnitude) per epoch estimated for ec2 g2
        120.1 mb estimated for model
        """


class Experiment082(object):
    """
    comments: like 63 except:
    (0) using pp.Rotator360PlusGeometry
    (1) using overlapping pools
    (2) adam
    (3) nn.WackyLearningRateAnnealer with reset_on_decay=validation

    results:

    """
    def __init__(p):
        p.name = 'e082'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 237
        p.epochs = 10000
        p.final_epochs = 40
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (77, 77)

        p.rng_seed = 13579

        p.learning_rule = nn.Adam(
            b1=0.9,
            b2=0.999,
            e=1e-8,
            lmbda=1 - 1e-8,
            learning_rate_annealer=nn.WackyLearningRateAnnealer(
                learning_rate=0.001,
                min_learning_rate=0.00001,
                patience=20,
                train_improvement_threshold=.995,
                valid_improvement_threshold=0.99995,
                reset_on_decay='validation'))

        p.activation = nn.relu

        p.batch_normalize = True

        p.l1_reg = 0.000
        p.l2_reg = 0.00005
        p.dropout_conv = 0.35
        p.dropout_hidd = 0.55

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360PlusGeometry

    def build_net(p, n):
        n.add_convolution(64, (7, 7), (3, 3), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(96, (5, 5), (3, 3), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(96, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_conv_pool(128, (3, 3), (3, 3), (2, 2), use_bias=False)
        n.merge_data_channel('geometry')
        n.add_batch_normalization()
        n.add_nonlinearity(p.activation)
        n.add_dropout(p.dropout_conv)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_logistic()

        """
        type | calcs | params |    neurons     |     output     | mults | adds | weights | biases | pool loss
        ------------------------------------------------------------------------------------------------------
        conv |   32m |     3t |   (64, 71, 71) |   (64, 35, 35) |   16m |  16m |      3t |    64  |
        conv |  295m |     2t |   (96, 31, 31) |   (96, 15, 15) |  148m | 148m |      2t |    96  |
        conv |   28m |   960  |   (96, 13, 13) |   (96, 13, 13) |   14m |  14m |    864  |    96  |
        conv |   27m |     1t |  (128, 11, 11) |  (128, 11, 11) |   13m |  13m |      1t |   128  |
        conv |   24m |     1t |    (128, 9, 9) |    (128, 4, 4) |   12m |  12m |      1t |   128  |
        full |   13m |     6m |           3072 |           3072 |    6m |   6m |      6m |     3t |
        full |   19m |     9m |           3072 |           3072 |    9m |   9m |      9m |     3t |
        ----------------------------------------
        total   437m      16m           463104

        7.29 mins (+/- 1 order of magnitude) per epoch estimated for ec2 g2
        120.1 mb estimated for model
        """



class Experiment081(object):
    """
    comments:
    like 68 except:
    (1) using pp.Rotator360PlusGeometry
    (1) adam
    (2) learning rate decay

    results:

    """
    def __init__(p):
        p.name = 'e081'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 237
        p.epochs = 10000
        p.final_epochs = 40
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (78, 78)

        p.rng_seed = 13579

        p.learning_rule = nn.Adam(
            b1=0.9,
            b2=0.999,
            e=1e-8,
            lmbda=1 - 1e-8,
            learning_rate_annealer=nn.LearningRateDecay(
                learning_rate=0.001,
                decay=.03))

        p.activation = nn.relu

        p.batch_normalize = True

        p.l1_reg = 0.000
        p.l2_reg = 0.00005
        p.dropout_conv = 0.45
        p.dropout_hidd = 0.6

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360PlusGeometry

    def build_net(p, n):
        n.add_convolution(72, (7, 7), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (5, 5), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(192, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_conv_pool(128, (3, 3), (2, 2), use_bias=False)
        n.merge_data_channel('geometry')
        n.add_batch_normalization()
        n.add_nonlinearity(p.activation)
        n.add_dropout(p.dropout_conv)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_logistic()


        """
        type | calcs | params |    neurons     |     output     | mults | adds | weights | biases | pool loss
        ------------------------------------------------------------------------------------------------------
        conv |   37m |     4t |   (72, 72, 72) |   (72, 36, 36) |   18m |  19m |      4t |    72  |
        conv |  472m |     3t |  (128, 32, 32) |  (128, 16, 16) |  236m | 236m |      3t |   128  |
        conv |   58m |     1t |  (128, 14, 14) |  (128, 14, 14) |   29m |  29m |      1t |   128  |
        conv |   64m |     2t |  (192, 12, 12) |  (192, 12, 12) |   32m |  32m |      2t |   192  |
        conv |   44m |     1t |  (128, 10, 10) |    (128, 5, 5) |   22m |  22m |      1t |   128  |
        full |   20m |    10m |           3072 |           3072 |   10m |  10m |     10m |     3t |
        full |   19m |     9m |           3072 |           3072 |    9m |   9m |      9m |     3t |
        full |   19m |     9m |           3072 |           3072 |    9m |   9m |      9m |     3t |
        ----------------------------------------
        total   732m      29m           579072

        12.20 mins (+/- 1 order of magnitude) per epoch estimated for ec2 g2
        219.2 mb estimated for model
        """


class Experiment080(object):
    """
    comments:
    like 68 except:
    (1) using pp.Rotator360PlusGeometry
    (2) using overlapping pools
    (1) adam
    (2) learning rate decay

    results:


    """
    def __init__(p):
        p.name = 'e080'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 237
        p.epochs = 10000
        p.final_epochs = 40
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (77, 77)

        p.rng_seed = 13579

        p.learning_rule = nn.Adam(
            b1=0.9,
            b2=0.999,
            e=1e-8,
            lmbda=1 - 1e-8,
            learning_rate_annealer=nn.LearningRateDecay(
                learning_rate=0.001,
                decay=.03))

        p.activation = nn.relu

        p.batch_normalize = True

        p.l1_reg = 0.000
        p.l2_reg = 0.00005
        p.dropout_conv = 0.45
        p.dropout_hidd = 0.6

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360PlusGeometry

    def build_net(p, n):
        n.add_convolution(72, (7, 7), (3, 3), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (5, 5), (3, 3), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(192, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_conv_pool(128, (3, 3), (3, 3), (2, 2), use_bias=False)
        n.merge_data_channel('geometry')
        n.add_batch_normalization()
        n.add_nonlinearity(p.activation)
        n.add_dropout(p.dropout_conv)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_logistic()


        """
        type | calcs | params |    neurons     |     output     | mults | adds | weights | biases | pool loss
        ------------------------------------------------------------------------------------------------------
        conv |   36m |     4t |   (72, 71, 71) |   (72, 35, 35) |   18m |  18m |      4t |    72  |
        conv |  443m |     3t |  (128, 31, 31) |  (128, 15, 15) |  221m | 222m |      3t |   128  |
        conv |   50m |     1t |  (128, 13, 13) |  (128, 13, 13) |   25m |  25m |      1t |   128  |
        conv |   54m |     2t |  (192, 11, 11) |  (192, 11, 11) |   27m |  27m |      2t |   192  |
        conv |   36m |     1t |    (128, 9, 9) |    (128, 4, 4) |   18m |  18m |      1t |   128  |
        full |   13m |     6m |           3072 |           3072 |    6m |   6m |      6m |     3t |
        full |   19m |     9m |           3072 |           3072 |    9m |   9m |      9m |     3t |
        full |   19m |     9m |           3072 |           3072 |    9m |   9m |      9m |     3t |
        ----------------------------------------
        total   668m      25m           550408

        11.14 mins (+/- 1 order of magnitude) per epoch estimated for ec2 g2
        192.2 mb estimated for model
        """


class Experiment079(object):
    """
    comments: like 63 except:
    (0) using pp.Rotator360PlusGeometry
    (1) adam
    (2) learning rate decay

    results:

    """
    def __init__(p):
        p.name = 'e079'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 237
        p.epochs = 10000
        p.final_epochs = 40
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (70, 70)

        p.rng_seed = 13579

        p.learning_rule = nn.Adam(
            b1=0.9,
            b2=0.999,
            e=1e-8,
            lmbda=1 - 1e-8,
            learning_rate_annealer=nn.LearningRateDecay(
                learning_rate=0.001,
                decay=.03))

        p.activation = nn.relu

        p.batch_normalize = True

        p.l1_reg = 0.000
        p.l2_reg = 0.00005
        p.dropout_conv = 0.35
        p.dropout_hidd = 0.55

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360PlusGeometry

    def build_net(p, n):
        n.add_convolution(64, (7, 7), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(96, (5, 5), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(96, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_conv_pool(128, (3, 3), (2, 2), use_bias=False)
        n.merge_data_channel('geometry')
        n.add_batch_normalization()
        n.add_nonlinearity(p.activation)
        n.add_dropout(p.dropout_conv)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_logistic()

        # type | calcs | params |    neurons     |     output     | mults | adds | weights | biases
        # ----------------------------------------------------------------------
        # conv |   26m |     3t |   (64, 64, 64) |   (64, 32, 32) |   13m |  13m |      3t |   64
        # conv |  241m |     2t |   (96, 28, 28) |   (96, 14, 14) |  120m | 120m |      2t |   96
        # conv |   24m |   960  |   (96, 12, 12) |   (96, 12, 12) |   12m |  12m |    864  |   96
        # conv |   22m |     1t |  (128, 10, 10) |  (128, 10, 10) |   11m |  11m |      1t |  128
        # conv |   19m |     1t |    (128, 8, 8) |    (128, 4, 4) |    9m |   9m |      1t |  128
        # full |   13m |     6m |           3072 |           3072 |    6m |   6m |      6m |    3t
        # full |   19m |     9m |           3072 |           3072 |    9m |   9m |      9m |    3t
        # ----------------------------------------
        # total   363m      16m           378368
        #
        # 6.05 mins (+/- 1 order of magnitude) per epoch estimated for ec2 g2
        # 120.1 mb estimated for model



class Experiment078(object):
    """
    comments: like 63 except:
    (0) using pp.Rotator360PlusGeometry
    (1) using overlapping pools
    (2) adam
    (3) learning rate decay

    results:

    """
    def __init__(p):
        p.name = 'e078'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 237
        p.epochs = 10000
        p.final_epochs = 40
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (77, 77)

        p.rng_seed = 13579

        p.learning_rule = nn.Adam(
            b1=0.9,
            b2=0.999,
            e=1e-8,
            lmbda=1 - 1e-8,
            learning_rate_annealer=nn.LearningRateDecay(
                learning_rate=0.001,
                decay=.03))

        p.activation = nn.relu

        p.batch_normalize = True

        p.l1_reg = 0.000
        p.l2_reg = 0.00005
        p.dropout_conv = 0.35
        p.dropout_hidd = 0.55

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360PlusGeometry

    def build_net(p, n):
        n.add_convolution(64, (7, 7), (3, 3), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(96, (5, 5), (3, 3), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(96, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_conv_pool(128, (3, 3), (3, 3), (2, 2), use_bias=False)
        n.merge_data_channel('geometry')
        n.add_batch_normalization()
        n.add_nonlinearity(p.activation)
        n.add_dropout(p.dropout_conv)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_logistic()

        """
        type | calcs | params |    neurons     |     output     | mults | adds | weights | biases | pool loss
        ------------------------------------------------------------------------------------------------------
        conv |   32m |     3t |   (64, 71, 71) |   (64, 35, 35) |   16m |  16m |      3t |    64  |
        conv |  295m |     2t |   (96, 31, 31) |   (96, 15, 15) |  148m | 148m |      2t |    96  |
        conv |   28m |   960  |   (96, 13, 13) |   (96, 13, 13) |   14m |  14m |    864  |    96  |
        conv |   27m |     1t |  (128, 11, 11) |  (128, 11, 11) |   13m |  13m |      1t |   128  |
        conv |   24m |     1t |    (128, 9, 9) |    (128, 4, 4) |   12m |  12m |      1t |   128  |
        full |   13m |     6m |           3072 |           3072 |    6m |   6m |      6m |     3t |
        full |   19m |     9m |           3072 |           3072 |    9m |   9m |      9m |     3t |
        ----------------------------------------
        total   437m      16m           463104

        7.29 mins (+/- 1 order of magnitude) per epoch estimated for ec2 g2
        120.1 mb estimated for model
        """


class Experiment077(object):
    """
    comments:
    like 68 except:
    (1) overlapping pools

    like 76 except:
    (1) pp.Rotator360 (instead of pp.Rotator360PlusGeometry)

    results: oops... forgot to set screen I guess.  Seems to have died.
    Decided not to rerun it.

    """
    def __init__(p):
        p.name = 'e077'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 237
        p.epochs = 10000
        p.final_epochs = 40
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (77, 77)

        p.rng_seed = 13579

        p.learning_rule = nn.Momentum
        p.learning_rate = .4
        p.min_learning_rate = 0.0003
        p.patience = 20
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.90
        p.activation = nn.relu

        p.batch_normalize = True

        p.l1_reg = 0.000
        p.l2_reg = 0.00005
        p.dropout_conv = 0.45
        p.dropout_hidd = 0.6

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360

    def build_net(p, n):
        n.add_convolution(72, (7, 7), (3, 3), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (5, 5), (3, 3), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(192, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (3, 3), (3, 3), (2, 2), batch_normalize=p.batch_normalize)
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_conv)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_logistic()


        """
        type | calcs | params |    neurons     |     output     | mults | adds | weights | biases | pool loss
        ------------------------------------------------------------------------------------------------------
        conv |   37m |     4t |   (72, 72, 72) |   (72, 36, 36) |   18m |  19m |      4t |    72  |
        conv |  472m |     3t |  (128, 32, 32) |  (128, 16, 16) |  236m | 236m |      3t |   128  |
        conv |   58m |     1t |  (128, 14, 14) |  (128, 14, 14) |   29m |  29m |      1t |   128  |
        conv |   64m |     2t |  (192, 12, 12) |  (192, 12, 12) |   32m |  32m |      2t |   192  |
        conv |   44m |     1t |  (128, 10, 10) |    (128, 5, 5) |   22m |  22m |      1t |   128  |
        full |   20m |    10m |           3072 |           3072 |   10m |  10m |     10m |     3t |
        full |   19m |     9m |           3072 |           3072 |    9m |   9m |      9m |     3t |
        full |   19m |     9m |           3072 |           3072 |    9m |   9m |      9m |     3t |
        ----------------------------------------
        total   732m      29m           579072

        12.20 mins (+/- 1 order of magnitude) per epoch estimated for ec2 g2
        219.2 mb estimated for model
        """


class Experiment076(object):
    """
    comments:
    like 68 except:
    (1) using pp.Rotator360PlusGeometry
    (2) using overlapping pools

    results:
    Amazon killed it an this was the only thing I could copy from the screen.

    Pretty much not exciting.  Not training nearly as well as 68 trains with
    Adam.

     455   | 24.1%, 0.826169 | 21.5%, 0.624453 * | 25.5%, 0.840706 |
     456   | 23.9%, 0.828213 | 21.1%, 0.620496 * |                 | frustration=1,
     457   | 24.3%, 0.829621 | 20.4%, 0.592010   |                 |
     458   | 24.1%, 0.825300 | 20.5%, 0.594062 * | 25.3%, 0.836774 |
     459   | 24.0%, 0.826488 | 21.0%, 0.607424 * |                 | frustration=1,
     460   | 23.8%, 0.830746 | 20.8%, 0.612851 * |                 | frustration=2,
     461   | 23.9%, 0.827300 | 21.1%, 0.610594 * |                 | frustration=3,
     462   | 24.3%, 0.823094 | 20.0%, 0.578284   | 25.3%, 0.830205 |
     463   | 24.2%, 0.827380 | 20.9%, 0.602608 * |                 | frustration=1,
     464   | 24.1%, 0.827180 | 21.1%, 0.606646 * |                 | frustration=2,
     465   | 24.2%, 0.828780 | 20.8%, 0.609586 * |                 | frustration=3,
     466   | 24.2%, 0.826625 | 20.6%, 0.598769 * |                 | frustration=4,
     467   | 24.1%, 0.825626 | 20.7%, 0.604052 * |                 | frustration=5,
     468   | 24.1%, 0.824750 | 21.0%, 0.603454 * |                 | frustration=6,
     469   | 24.3%, 0.827006 | 19.8%, 0.577059   |                 | frustration=7,
     470   | 23.9%, 0.827054 | 19.7%, 0.572503   |                 |
     471   | 23.9%, 0.826804 | 20.4%, 0.593093 * |                 | frustration=1,
     472   | 24.2%, 0.824606 | 20.6%, 0.600088 * |                 | frustration=2,
     473   | 24.2%, 0.826519 | 20.6%, 0.593191 * |                 | frustration=3,
     474   | 24.1%, 0.828630 | 20.0%, 0.569889   |                 | frustration=4,
     475   | 24.1%, 0.827687 | 20.3%, 0.589253 * |                 | frustration=5,
     476   | 24.1%, 0.827505 | 19.7%, 0.567085   |                 |
     477   | 24.1%, 0.828112 | 20.2%, 0.590267 * |                 | frustration=1,
     478   | 24.0%, 0.830645 | 20.2%, 0.585123 * |                 | frustration=2,
     479   | 23.9%, 0.825412 | 20.1%, 0.586250 * |                 | frustration=3,
     480   | 23.9%, 0.828423 | 20.4%, 0.590023 * |                 | frustration=4,
     481   | 23.9%, 0.829589 | 19.5%, 0.566369   |                 | frustration=5,
     482   | 23.6%, 0.824802 | 20.0%, 0.579110 * |                 | frustration=6,
     483   | 24.0%, 0.826887 | 19.6%, 0.570167 * |                 | frustration=7,
     484   | 24.0%, 0.831219 | 20.0%, 0.575778 * |                 | frustration=8,
     485   | 24.1%, 0.828134 | 20.0%, 0.581712 * |                 | frustration=9,
     486   | 24.1%, 0.829439 | 20.2%, 0.584214 * |                 | frustration=10,
     487   | 24.0%, 0.829518 | 19.7%, 0.569412 * |                 | frustration=11,
     488   | 24.1%, 0.830230 | 19.7%, 0.564028   |                 |
     489   | 24.1%, 0.835512 | 20.2%, 0.579022 * |                 | frustration=1,
     490   | 24.0%, 0.831839 | 20.0%, 0.575702 * |                 | frustration=2,
     491   | 23.9%, 0.830113 | 19.4%, 0.562852   |                 | frustration=3,
     492   | 24.1%, 0.828733 | 20.2%, 0.579879 * |                 | frustration=4,
     493   | 24.1%, 0.834073 | 19.8%, 0.573303 * |                 | frustration=5,
     494   | 24.5%, 0.835379 | 19.8%, 0.574863 * |                 | frustration=6,
     495   | 24.1%, 0.833082 | 19.2%, 0.555339   |                 |
     496   | 24.0%, 0.833696 | 19.6%, 0.561924 * |                 | frustration=1,
     497   | 24.2%, 0.832577 | 19.6%, 0.566149 * |                 | frustration=2,
     498   | 24.1%, 0.833318 | 19.6%, 0.567233 * |                 | frustration=3,
     499   | 23.9%, 0.832430 | 19.3%, 0.554101   |                 | frustration=4,
     500   | 23.9%, 0.832289 | 19.9%, 0.569674 * |                 | frustration=5,
     501   | 24.1%, 0.832948 | 20.1%, 0.577642 * |                 | frustration=6,
     502   | 23.8%, 0.828979 | 19.7%, 0.567558 * |                 | frustration=7,
     503   | 24.3%, 0.833198 | 19.9%, 0.570320 * |                 | frustration=8,
     504   | 24.1%, 0.829881 | 19.2%, 0.552231   |                 |
     505   | 24.0%, 0.837194 | 19.2%, 0.547675   |                 |
     506   | 24.0%, 0.836279 | 19.8%, 0.563982 * |                 | frustration=1,
     507   | 23.9%, 0.836832 | 19.3%, 0.546935   |                 | frustration=2,



    """
    def __init__(p):
        p.name = 'e076'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 237
        p.epochs = 10000
        p.final_epochs = 40
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (77, 77)

        p.rng_seed = 13579

        p.learning_rule = nn.Momentum
        p.learning_rate = .4
        p.min_learning_rate = 0.0003
        p.patience = 20
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.90
        p.activation = nn.relu

        p.batch_normalize = True

        p.l1_reg = 0.000
        p.l2_reg = 0.00005
        p.dropout_conv = 0.45
        p.dropout_hidd = 0.6

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360PlusGeometry

    def build_net(p, n):
        n.add_convolution(72, (7, 7), (3, 3), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (5, 5), (3, 3), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(192, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_conv_pool(128, (3, 3), (3, 3), (2, 2), use_bias=False)
        n.merge_data_channel('geometry')
        n.add_batch_normalization()
        n.add_nonlinearity(p.activation)
        n.add_dropout(p.dropout_conv)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_logistic()


        """
        type | calcs | params |    neurons     |     output     | mults | adds | weights | biases | pool loss
        ------------------------------------------------------------------------------------------------------
        conv |   36m |     4t |   (72, 71, 71) |   (72, 35, 35) |   18m |  18m |      4t |    72  |
        conv |  443m |     3t |  (128, 31, 31) |  (128, 15, 15) |  221m | 222m |      3t |   128  |
        conv |   50m |     1t |  (128, 13, 13) |  (128, 13, 13) |   25m |  25m |      1t |   128  |
        conv |   54m |     2t |  (192, 11, 11) |  (192, 11, 11) |   27m |  27m |      2t |   192  |
        conv |   36m |     1t |    (128, 9, 9) |    (128, 4, 4) |   18m |  18m |      1t |   128  |
        full |   13m |     6m |           3072 |           3072 |    6m |   6m |      6m |     3t |
        full |   19m |     9m |           3072 |           3072 |    9m |   9m |      9m |     3t |
        full |   19m |     9m |           3072 |           3072 |    9m |   9m |      9m |     3t |
        ----------------------------------------
        total   668m      25m           550408

        11.14 mins (+/- 1 order of magnitude) per epoch estimated for ec2 g2
        192.2 mb estimated for model
        """


class Experiment075(object):
    """
    comments: like 63 except:
    (0) using pp.Rotator360PlusGeometry
    (1) using overlapping pools

    like 69 but:
    (0) using pp.Rotator360PlusGeometry

    results:
    When combined with strides geometry doesn't seem to do anything useful

    e075:
    318   | 24.3%, 0.806964 | 20.0%, 0.584183 * | 25.9%, 0.831931 |

    e069:
    247   | 24.3%, 0.805532 | 20.3%, 0.595169 * | 24.3%, 0.829145 |

    e063:
    323   | 24.1%, 0.811939 | 20.2%, 0.585712 * | 25.2%, 0.828347 |



    """
    def __init__(p):
        p.name = 'e075'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 237
        p.epochs = 10000
        p.final_epochs = 40
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (77, 77)

        p.rng_seed = 13579

        p.learning_rule = nn.Momentum
        p.learning_rate = .4
        p.min_learning_rate = 0.0003
        p.patience = 20
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.90
        p.activation = nn.relu

        p.batch_normalize = True

        p.l1_reg = 0.000
        p.l2_reg = 0.00005
        p.dropout_conv = 0.35
        p.dropout_hidd = 0.55

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360PlusGeometry

    def build_net(p, n):
        n.add_convolution(64, (7, 7), (3, 3), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(96, (5, 5), (3, 3), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(96, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_conv_pool(128, (3, 3), (3, 3), (2, 2), use_bias=False)
        n.merge_data_channel('geometry')
        n.add_batch_normalization()
        n.add_nonlinearity(p.activation)
        n.add_dropout(p.dropout_conv)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_logistic()

        """
        type | calcs | params |    neurons     |     output     | mults | adds | weights | biases | pool loss
        ------------------------------------------------------------------------------------------------------
        conv |   32m |     3t |   (64, 71, 71) |   (64, 35, 35) |   16m |  16m |      3t |    64  |
        conv |  295m |     2t |   (96, 31, 31) |   (96, 15, 15) |  148m | 148m |      2t |    96  |
        conv |   28m |   960  |   (96, 13, 13) |   (96, 13, 13) |   14m |  14m |    864  |    96  |
        conv |   27m |     1t |  (128, 11, 11) |  (128, 11, 11) |   13m |  13m |      1t |   128  |
        conv |   24m |     1t |    (128, 9, 9) |    (128, 4, 4) |   12m |  12m |      1t |   128  |
        full |   13m |     6m |           3072 |           3072 |    6m |   6m |      6m |     3t |
        full |   19m |     9m |           3072 |           3072 |    9m |   9m |      9m |     3t |
        ----------------------------------------
        total   437m      16m           463104

        7.29 mins (+/- 1 order of magnitude) per epoch estimated for ec2 g2
        120.1 mb estimated for model
        """


class Experiment074(object):
    """
    comments:
    like 68 except:
    (1) using pp.Rotator360PlusGeometry

    results:
    seems to be doing roughly the same as 68 by epoch 085

    e074:
    085   | 28.2%, 0.939886 | 32.5%, 1.022044   | 30.5%, 0.996703 |

    e068:
    085   | 28.2%, 0.931830 | 32.8%, 1.035783 * | 30.1%, 1.004930 | momentum=0.681,


    """
    def __init__(p):
        p.name = 'e074'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 237
        p.epochs = 10000
        p.final_epochs = 40
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (78, 78)

        p.rng_seed = 13579

        p.learning_rule = nn.Momentum
        p.learning_rate = .4
        p.min_learning_rate = 0.0003
        p.patience = 20
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.90
        p.activation = nn.relu

        p.batch_normalize = True

        p.l1_reg = 0.000
        p.l2_reg = 0.00005
        p.dropout_conv = 0.45
        p.dropout_hidd = 0.6

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360PlusGeometry

    def build_net(p, n):
        n.add_convolution(72, (7, 7), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (5, 5), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(192, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_conv_pool(128, (3, 3), (2, 2), use_bias=False)
        n.merge_data_channel('geometry')
        n.add_batch_normalization()
        n.add_nonlinearity(p.activation)
        n.add_dropout(p.dropout_conv)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_logistic()


        """
        type | calcs | params |    neurons     |     output     | mults | adds | weights | biases | pool loss
        ------------------------------------------------------------------------------------------------------
        conv |   37m |     4t |   (72, 72, 72) |   (72, 36, 36) |   18m |  19m |      4t |    72  |
        conv |  472m |     3t |  (128, 32, 32) |  (128, 16, 16) |  236m | 236m |      3t |   128  |
        conv |   58m |     1t |  (128, 14, 14) |  (128, 14, 14) |   29m |  29m |      1t |   128  |
        conv |   64m |     2t |  (192, 12, 12) |  (192, 12, 12) |   32m |  32m |      2t |   192  |
        conv |   44m |     1t |  (128, 10, 10) |    (128, 5, 5) |   22m |  22m |      1t |   128  |
        full |   20m |    10m |           3072 |           3072 |   10m |  10m |     10m |     3t |
        full |   19m |     9m |           3072 |           3072 |    9m |   9m |      9m |     3t |
        full |   19m |     9m |           3072 |           3072 |    9m |   9m |      9m |     3t |
        ----------------------------------------
        total   732m      29m           579072

        12.20 mins (+/- 1 order of magnitude) per epoch estimated for ec2 g2
        219.2 mb estimated for model
        """


class Experiment073(object):
    """
    comments: like 63 except:
    (0) using pp.Rotator360PlusGeometry

    results:

    """
    def __init__(p):
        p.name = 'e073'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 237
        p.epochs = 10000
        p.final_epochs = 40
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (70, 70)

        p.rng_seed = 13579

        p.learning_rule = nn.Momentum
        p.learning_rate = .4
        p.min_learning_rate = 0.0003
        p.patience = 20
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.90
        p.activation = nn.relu

        p.batch_normalize = True

        p.l1_reg = 0.000
        p.l2_reg = 0.00005
        p.dropout_conv = 0.35
        p.dropout_hidd = 0.55

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360PlusGeometry

    def build_net(p, n):
        n.add_convolution(64, (7, 7), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(96, (5, 5), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(96, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_conv_pool(128, (3, 3), (2, 2), use_bias=False)
        n.merge_data_channel('geometry')
        n.add_batch_normalization()
        n.add_nonlinearity(p.activation)
        n.add_dropout(p.dropout_conv)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_logistic()

        # type | calcs | params |    neurons     |     output     | mults | adds | weights | biases
        # ----------------------------------------------------------------------
        # conv |   26m |     3t |   (64, 64, 64) |   (64, 32, 32) |   13m |  13m |      3t |   64
        # conv |  241m |     2t |   (96, 28, 28) |   (96, 14, 14) |  120m | 120m |      2t |   96
        # conv |   24m |   960  |   (96, 12, 12) |   (96, 12, 12) |   12m |  12m |    864  |   96
        # conv |   22m |     1t |  (128, 10, 10) |  (128, 10, 10) |   11m |  11m |      1t |  128
        # conv |   19m |     1t |    (128, 8, 8) |    (128, 4, 4) |    9m |   9m |      1t |  128
        # full |   13m |     6m |           3072 |           3072 |    6m |   6m |      6m |    3t
        # full |   19m |     9m |           3072 |           3072 |    9m |   9m |      9m |    3t
        # ----------------------------------------
        # total   363m      16m           378368
        #
        # 6.05 mins (+/- 1 order of magnitude) per epoch estimated for ec2 g2
        # 120.1 mb estimated for model



class Experiment072(object):
    """
    comments: like 63 but:
    (0) Adam optimizer

    like e071 but:
    (1) lower lambda

    results:
    Compared to adam with lower lambda (e071) Seems to have speed up learning in the very beginning a little bit
    but then slowed down learning later.

    I had meant to see how it would
    do with a learning rate decrease but the learning minimum learning rate
    was set too high so it stopped training without trying a lower rate.

    So I manually restarted training with the lower rate and it immediately
    made a pretty significant improvement over it's best validation score
    from the initial training.  But... I used the same starting lambda so
    that actually changes the training it was receiving pretty significantly...

    python
    >>> from examples.plankton import plankton
    >>> images, classes, file_indices, labels = plankton.build_training_set()
    >>> data = (images, classes, file_indices)
    >>> import experiment
    >>> e = experiment.Experiment()
    >>> import preprocessors as pp
    >>> data = pp.split_training_data(data, e.batch_size, e.train_pct, e.valid_pct, e.test_pct)
    >>> import nn
    >>> net = nn.load('e072_best_net.pkl')
    >>> net.preprocessor.set_data(data)
    >>> updater = nn.Adam(0.0001, 0.00001, 20, 0.995, e.b1, e.b2, e.e, e.lmbda)
    >>> net.train(updater, e.epochs, e._final_epochs, e.l1_reg, e.l2_reg)
    """

    def __init__(p):
        p.name = 'e072'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 237
        p.epochs = 10000
        p.final_epochs = 40
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (70, 70)

        p.rng_seed = 13579

        p.learning_rule = nn.Adam

        p.b1 = 0.9
        p.b2 = 0.999
        p.e = 1e-8
        p.lmbda = 0.9999
        p.learning_rate = 0.001

        p.min_learning_rate = 0.0003
        p.patience = 20
        p.improvement_threshold = 0.995
        p.activation = nn.relu

        p.batch_normalize = True

        p.l1_reg = 0.000
        p.l2_reg = 0.00005
        p.dropout_conv = 0.35
        p.dropout_hidd = 0.55

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360

    def build_net(p, n):
        n.add_convolution(64, (7, 7), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(96, (5, 5), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(96, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (3, 3), (2, 2), batch_normalize=p.batch_normalize)
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_conv)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_hidd)
        n.add_logistic()

        # type | calcs | params |    neurons     |     output     | mults | adds | weights | biases
        # ----------------------------------------------------------------------
        # conv |   26m |     3t |   (64, 64, 64) |   (64, 32, 32) |   13m |  13m |      3t |   64
        # conv |  241m |     2t |   (96, 28, 28) |   (96, 14, 14) |  120m | 120m |      2t |   96
        # conv |   24m |   960  |   (96, 12, 12) |   (96, 12, 12) |   12m |  12m |    864  |   96
        # conv |   22m |     1t |  (128, 10, 10) |  (128, 10, 10) |   11m |  11m |      1t |  128
        # conv |   19m |     1t |    (128, 8, 8) |    (128, 4, 4) |    9m |   9m |      1t |  128
        # full |   13m |     6m |           3072 |           3072 |    6m |   6m |      6m |    3t
        # full |   19m |     9m |           3072 |           3072 |    9m |   9m |      9m |    3t
        # ----------------------------------------
        # total   363m      16m           378368
        #
        # 6.05 mins (+/- 1 order of magnitude) per epoch estimated for ec2 g2
        # 120.1 mb estimated for model


class Experiment071(object):
    """
    comments: like 63 but:
    (0) Adam optimizer

    Testing fixed version of adam optimizer based on skaae

    results:
    Training slower than momentum so far, but interestingly, when comparing
    different epochs with the same validation score, adam has much higher
    training score.  So it is generalizing significantly better.  I guess that
    is not too surprising since I've had questions about my learning rate annealing
    and the wisdom of choose the best training score on learning rate changes.
    I wonder if that is why.  At any rate... very interesting.  Adam might
    be worth experimenting with more...


    I had meant to see how it would
    do with a learning rate decrease but the learning minimum learning rate
    was set too high so it stopped training without trying a lower rate.

    So I manually restarted training with the lower rate and it immediately
    made a pretty significant improvement over it's best validation score
    from the initial training.  But... I used the same starting lambda so
    that actually changes the training it was receiving pretty significantly...

    python
    >>> from examples.plankton import plankton
    >>> images, classes, file_indices, labels = plankton.build_training_set()
    >>> data = (images, classes, file_indices)
    >>> import experiment
    >>> e = experiment.Experiment()
    >>> import preprocessors as pp
    >>> data = pp.split_training_data(data, e.batch_size, e.train_pct, e.valid_pct, e.test_pct)
    >>> import nn
    >>> net = nn.load('e071_best_net.pkl')
    >>> net.preprocessor.set_data(data)
    >>> updater = nn.Adam(0.0001, 0.00001, 20, 0.995, e.b1, e.b2, e.e, e.lmbda)
    >>> net.train(updater, e.epochs, e._final_epochs, e.l1_reg, e.l2_reg)
    """
    def __init__(p):
        p.name = 'e071'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 237
        p.epochs = 10000
        p.final_epochs = 40
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (70, 70)

        p.rng_seed = 13579

        p.learning_rule = nn.Adam

        p.b1 = 0.9
        p.b2 = 0.999
        p.e = 1e-8
        p.lmbda = 1 - 1e-8
        p.learning_rate = 0.001

        p.min_learning_rate = 0.0003
        p.patience = 20
        p.improvement_threshold = 0.995
        p.activation = nn.relu

        p.batch_normalize = True

        p.l1_reg = 0.000
        p.l2_reg = 0.00005
        p.dropout_conv = 0.35
        p.dropout_hidd = 0.55

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360

    def build_net(p, n):
        n.add_convolution(64, (7, 7), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(96, (5, 5), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(96, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (3, 3), (2, 2), batch_normalize=p.batch_normalize)
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_conv)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_hidd)
        n.add_logistic()

        # type | calcs | params |    neurons     |     output     | mults | adds | weights | biases
        # ----------------------------------------------------------------------
        # conv |   26m |     3t |   (64, 64, 64) |   (64, 32, 32) |   13m |  13m |      3t |   64
        # conv |  241m |     2t |   (96, 28, 28) |   (96, 14, 14) |  120m | 120m |      2t |   96
        # conv |   24m |   960  |   (96, 12, 12) |   (96, 12, 12) |   12m |  12m |    864  |   96
        # conv |   22m |     1t |  (128, 10, 10) |  (128, 10, 10) |   11m |  11m |      1t |  128
        # conv |   19m |     1t |    (128, 8, 8) |    (128, 4, 4) |    9m |   9m |      1t |  128
        # full |   13m |     6m |           3072 |           3072 |    6m |   6m |      6m |    3t
        # full |   19m |     9m |           3072 |           3072 |    9m |   9m |      9m |    3t
        # ----------------------------------------
        # total   363m      16m           378368
        #
        # 6.05 mins (+/- 1 order of magnitude) per epoch estimated for ec2 g2
        # 120.1 mb estimated for model


class Experiment070(object):
    """
    comments: like 63 but:
    (0) Adam optimizer

    Testing with two versions: nn.Adam and nn.AdamDeprecated
    In the first few epochs nn.Adam is showing substantial
    improvements over e063 (trained with Momentum).
    nn.AdamDeprecated might be showing slight improvements over e063 but
    doesn't look nearly as promising as nn.Adam

    results:
    Starts off training much faster than momentum but then slows down a lot.

    e070:
    000   | 59.2%, 2.162153 | 72.2%, 2.866029   | 60.2%, 2.187642 | train time: 238.4, test time: 8.7, learning rate=0.002000,
    001   | 54.0%, 1.852738 | 60.4%, 2.142764   | 53.8%, 1.865528 |
    140   | 28.5%, 0.928359 | 33.1%, 1.068966 * | 30.9%, 0.982257 |

    e063:
    000   | 64.1%, 2.465194 | 78.5%, 3.314696   | 63.8%, 2.481129 | train time: 235.0, test time: 7.6, learning rate=0.400000,
    001   | 61.1%, 2.250665 | 66.2%, 2.502658   | 58.2%, 2.199989 |
    134   | 25.8%, 0.841465 | 26.1%, 0.793165 * | 26.6%, 0.875059 |

    Don't see much point in continuing.  killing it
    """
    def __init__(p):
        p.name = 'e070'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 237
        p.epochs = 10000
        p.final_epochs = 40
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (70, 70)

        p.rng_seed = 13579

        p.learning_rule = nn.AdamNewmu

        p.b1 = 0.1
        p.b2 = 0.001
        p.e = 1e-8
        p.learning_rate = 0.002

        # p.b1 = 0.9
        # p.b2 = 0.999
        # p.e = 1e-8
        # p.lmbda = 1 - 1e-8
        # p.learning_rate = 0.001

        p.min_learning_rate = 0.0003
        p.patience = 20
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.90
        p.activation = nn.relu

        p.batch_normalize = True

        p.l1_reg = 0.000
        p.l2_reg = 0.00005
        p.dropout_conv = 0.35
        p.dropout_hidd = 0.55

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360

    def build_net(p, n):
        n.add_convolution(64, (7, 7), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(96, (5, 5), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(96, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (3, 3), (2, 2), batch_normalize=p.batch_normalize)
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_conv)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_hidd)
        n.add_logistic()

        # type | calcs | params |    neurons     |     output     | mults | adds | weights | biases
        # ----------------------------------------------------------------------
        # conv |   26m |     3t |   (64, 64, 64) |   (64, 32, 32) |   13m |  13m |      3t |   64
        # conv |  241m |     2t |   (96, 28, 28) |   (96, 14, 14) |  120m | 120m |      2t |   96
        # conv |   24m |   960  |   (96, 12, 12) |   (96, 12, 12) |   12m |  12m |    864  |   96
        # conv |   22m |     1t |  (128, 10, 10) |  (128, 10, 10) |   11m |  11m |      1t |  128
        # conv |   19m |     1t |    (128, 8, 8) |    (128, 4, 4) |    9m |   9m |      1t |  128
        # full |   13m |     6m |           3072 |           3072 |    6m |   6m |      6m |    3t
        # full |   19m |     9m |           3072 |           3072 |    9m |   9m |      9m |    3t
        # ----------------------------------------
        # total   363m      16m           378368
        #
        # 6.05 mins (+/- 1 order of magnitude) per epoch estimated for ec2 g2
        # 120.1 mb estimated for model


class Experiment069(object):
    """
    comments: like 63 but:
    (1) overlapping pools (3, 3) with (2, 2) strides
    (2) bigger images (chosen to eliminate pooling information loss)

    results:
    looks like it makes a pretty significant positive impact
    e069:
    247   | 24.3%, 0.805532 | 20.3%, 0.595169 * | 24.3%, 0.829145 |

    e063:
    323   | 24.1%, 0.811939 | 20.2%, 0.585712 * | 25.2%, 0.828347 |

    """
    def __init__(p):
        p.name = 'e069'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 237
        p.epochs = 10000
        p.final_epochs = 40
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (77, 77)

        p.rng_seed = 13579

        p.learning_rule = nn.Momentum
        p.learning_rate = .4
        p.min_learning_rate = 0.0003
        p.patience = 20
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.90
        p.activation = nn.relu

        p.batch_normalize = True

        p.l1_reg = 0.000
        p.l2_reg = 0.00005
        p.dropout_conv = 0.35
        p.dropout_hidd = 0.55

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360

    def build_net(p, n):
        n.add_convolution(64, (7, 7), (3, 3), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(96, (5, 5), (3, 3), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(96, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (3, 3), (3, 3), (2, 2), batch_normalize=p.batch_normalize)
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_conv)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_hidd)
        n.add_logistic()

        """
        type | calcs | params |    neurons     |     output     | mults | adds | weights | biases | pool loss
        ------------------------------------------------------------------------------------------------------
        conv |   32m |     3t |   (64, 71, 71) |   (64, 35, 35) |   16m |  16m |      3t |    64  |
        conv |  295m |     2t |   (96, 31, 31) |   (96, 15, 15) |  148m | 148m |      2t |    96  |
        conv |   28m |   960  |   (96, 13, 13) |   (96, 13, 13) |   14m |  14m |    864  |    96  |
        conv |   27m |     1t |  (128, 11, 11) |  (128, 11, 11) |   13m |  13m |      1t |   128  |
        conv |   24m |     1t |    (128, 9, 9) |    (128, 4, 4) |   12m |  12m |      1t |   128  |
        full |   13m |     6m |           3072 |           3072 |    6m |   6m |      6m |     3t |
        full |   19m |     9m |           3072 |           3072 |    9m |   9m |      9m |     3t |
        ----------------------------------------
        total   437m      16m           463104

        7.29 mins (+/- 1 order of magnitude) per epoch estimated for ec2 g2
        120.1 mb estimated for model
        """


class Experiment068(object):
    """
    comments:
    like 63 except:
    (1) bigger images (70-> 78)
    (2) more filters in conv layers (64, 96, 96, 128, 128 -> 72, 128, 128, 192, 128)
    (3) one more hidden layer (2 -> 3)
    (4) higher dropout (0.35, 0.55 -> 0.4, 0.6)

    results:
    At epoch 186 its validation performance is nearly exactly the same as e063,
    but it's training performance is worse.  So it is generalizing better
    than e063.  If this doesn't result in improvement over e063 maybe I should
    run it again with slightly lower dropout.

    Looks like it is set to beat e063.  It's approaching e063 high score but
    with significantly less divergence and at an earlier training rate.
    284   | 24.9%, 0.818490 | 25.3%, 0.764402 * | 26.0%, 0.854372 |


    """
    def __init__(p):
        p.name = 'e068'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 237
        p.epochs = 10000
        p.final_epochs = 40
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (78, 78)

        p.rng_seed = 13579

        p.learning_rule = nn.Momentum
        p.learning_rate = .4
        p.min_learning_rate = 0.0003
        p.patience = 20
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.90
        p.activation = nn.relu

        p.batch_normalize = True

        p.l1_reg = 0.000
        p.l2_reg = 0.00005
        p.dropout_conv = 0.45
        p.dropout_hidd = 0.6

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360

    def build_net(p, n):
        n.add_convolution(72, (7, 7), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (5, 5), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(192, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (3, 3), (2, 2), batch_normalize=p.batch_normalize)
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_conv)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_logistic()


        """
        type | calcs | params |    neurons     |     output     | mults | adds | weights | biases | pool loss
        ------------------------------------------------------------------------------------------------------
        conv |   37m |     4t |   (72, 72, 72) |   (72, 36, 36) |   18m |  19m |      4t |    72  |
        conv |  472m |     3t |  (128, 32, 32) |  (128, 16, 16) |  236m | 236m |      3t |   128  |
        conv |   58m |     1t |  (128, 14, 14) |  (128, 14, 14) |   29m |  29m |      1t |   128  |
        conv |   64m |     2t |  (192, 12, 12) |  (192, 12, 12) |   32m |  32m |      2t |   192  |
        conv |   44m |     1t |  (128, 10, 10) |    (128, 5, 5) |   22m |  22m |      1t |   128  |
        full |   20m |    10m |           3072 |           3072 |   10m |  10m |     10m |     3t |
        full |   19m |     9m |           3072 |           3072 |    9m |   9m |      9m |     3t |
        full |   19m |     9m |           3072 |           3072 |    9m |   9m |      9m |     3t |
        ----------------------------------------
        total   732m      29m           579072

        12.20 mins (+/- 1 order of magnitude) per epoch estimated for ec2 g2
        219.2 mb estimated for model
        """


class Experiment067(object):
    """
    comments:
    compared to 66:
    (1) adding geometric data instead of shape data
    (2) adding geometric data directly before batch norm layer
    (3) adding geometric data only once right after convolutions

    compared to 65:
    (1) extra conv layer
    (2) slightly bigger image
    (1) adding geometric data instead of shape data
    (2) adding geometric data directly before batch norm layer
    (3) adding geometric data only once right after convolutions

    compared to 64:
    (1) extra conv layer
    (2) slightly bigger image
    (3) higher dropout
    (1) adding geometric data instead of shape data
    (2) adding geometric data directly before batch norm layer
    (3) adding geometric data only once right after convolutions

    compared to 63:
    (1) no pooling on last layer
    (2) extra conv layer
    (3) slightly bigger image
    (4) higher dropout
    (1) adding geometric data instead of shape data
    (2) adding geometric data directly before batch norm layer
    (3) adding geometric data only once right after convolutions


    results:
    So far, seems to be doing almost identically to e066, upon which it is based.
    Maybe seeing hints of a very slight improvement in generalization.  But
    I'll have to wait on that.

    MUCH SLOWER THAN 66 EVEN THOUGH IT IS ALMOST IDENTICAL.  I would have expected
    it to be very slightly faster, but instead seems to be taking twice the time.
    A bit mysterious.

    """
    def __init__(p):
        p.name = 'e067'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 237
        p.epochs = 10000
        p.final_epochs = 20
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (74, 74)

        p.rng_seed = 13579

        p.learning_rule = nn.Momentum
        p.learning_rate = .4
        p.min_learning_rate = 0.0003
        p.patience = 20
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.90
        p.activation = nn.relu

        p.l1_reg = 0.000
        p.l2_reg = 0.00005
        p.dropout_conv = 0.4
        p.dropout_hidd = 0.6

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360PlusGeometry

    def build_net(p, n):
        n.add_convolution(64, (7, 7), (2, 2), batch_normalize=True)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(96, (5, 5), (2, 2), batch_normalize=True)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(96, (3, 3), (1, 1), batch_normalize=True)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (3, 3), (1, 1), batch_normalize=True)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (3, 3), (1, 1), batch_normalize=True)
        n.add_dropout(p.dropout_conv)
        n.add_conv_pool(128, (3, 3), (1, 1), use_bias=False)
        n.merge_data_channel('geometry')
        n.add_batch_normalization()
        n.add_nonlinearity(p.activation)
        n.add_dropout(p.dropout_conv)
        n.add_hidden(3072, batch_normalize=True)
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(3072, batch_normalize=True)
        n.add_dropout(p.dropout_hidd)
        n.add_logistic()


class Experiment066(object):
    """
    comments:
    compared to 65:
    (1) extra conv layer
    (2) slightly bigger image

    running 65 as a baseline.  Compare to that primarily.  (So if 65 doesn't
    do any better than 64 or 63, but 66 does better than 65 that might indicate
    that I should keep experimenting with extra conv layers and bigger images.)

    compared to 64:
    (1) extra conv layer
    (2) slightly bigger image
    (3) higher dropout

    compared to 63:
    (1) no pooling on last layer
    (2) extra conv layer
    (3) slightly bigger image
    (4) higher dropout


    results:
    Doesn't seem to be doing any better than 65 so far
    Ended up with a slightly lower validation score a little bit earlier (epoch 280
    compared to epoch 323) and somewhat higher training score.  So maybe
    it was generalizing a little bit better.  If I had more time I would definitely
    experiment more with additional conv layers.

    """
    def __init__(p):
        p.name = 'e066'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 237
        p.epochs = 10000
        p.final_epochs = 20
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (74, 74)

        p.rng_seed = 13579

        p.learning_rule = nn.Momentum
        p.learning_rate = .4
        p.min_learning_rate = 0.0003
        p.patience = 20
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.90
        p.activation = nn.relu

        p.batch_normalize = True

        p.l1_reg = 0.000
        p.l2_reg = 0.00005
        p.dropout_conv = 0.4
        p.dropout_hidd = 0.6

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360

    def build_net(p, n):
        n.add_convolution(64, (7, 7), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(96, (5, 5), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(96, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_conv)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_hidd)
        n.add_logistic()

        """
        type | calcs | params |    neurons     |     output     | mults | adds | weights | biases
        ----------------------------------------------------------------------
        conv |   44m |     5t |   (96, 68, 68) |   (96, 34, 34) |   22m |  22m |      5t |   96
        conv |  415m |     2t |   (96, 30, 30) |   (96, 15, 15) |  207m | 207m |      2t |   96
        conv |   28m |   960  |   (96, 13, 13) |   (96, 13, 13) |   14m |  14m |    864  |   96
        conv |   27m |     1t |  (128, 11, 11) |  (128, 11, 11) |   13m |  13m |      1t |  128
        conv |   24m |     1t |    (128, 9, 9) |    (128, 9, 9) |   12m |  12m |      1t |  128
        conv |   14m |     1t |    (128, 7, 7) |    (128, 7, 7) |    7m |   7m |      1t |  128
        full |   39m |    19m |           3072 |           3072 |   19m |  19m |     19m |    3t
        full |   19m |     9m |           3072 |           3072 |    9m |   9m |      9m |    3t
        ----------------------------------------
        total   609m      29m           584800

        10.16 mins (+/- 1 order of magnitude) per epoch estimated for ec2 g2
        219.1 mb estimated for model
        """


class Experiment065(object):
    """
    comments:
    (1) no pooling on last layer
    (0) slightly bigger images
    (1) more patience
    (3) bigger hidden layers

    like 64 except:
    (1) higher dropout

    Compare to 64 and 63.  If it looks like it is overfitting more than
    63 but less than 64 might be worth trying more dropout, on this same model.

    like 63 except:
    (1) no pooling on last layer
    (2) higher dropout

    results:
    overfitting and performing slightly worse that 63.
    performing slightly worse than 64, but overfitting significantly less
    """
    def __init__(p):
        p.name = 'e065'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 237
        p.epochs = 10000
        p.final_epochs = 20
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (70, 70)

        p.rng_seed = 13579

        p.learning_rule = nn.Momentum
        p.learning_rate = .4
        p.min_learning_rate = 0.0003
        p.patience = 20
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.90
        p.activation = nn.relu

        p.batch_normalize = True

        p.l1_reg = 0.000
        p.l2_reg = 0.00005
        p.dropout_conv = 0.4
        p.dropout_hidd = 0.6

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360

    def build_net(p, n):
        n.add_convolution(64, (7, 7), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(96, (5, 5), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(96, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_conv)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_hidd)
        n.add_logistic()

        """
        type | calcs | params |    neurons     |     output     | mults | adds | weights | biases
        ----------------------------------------------------------------------
        conv |   26m |     3t |   (64, 64, 64) |   (64, 32, 32) |   13m |  13m |      3t |   64
        conv |  241m |     2t |   (96, 28, 28) |   (96, 14, 14) |  120m | 120m |      2t |   96
        conv |   24m |   960  |   (96, 12, 12) |   (96, 12, 12) |   12m |  12m |    864  |   96
        conv |   22m |     1t |  (128, 10, 10) |  (128, 10, 10) |   11m |  11m |      1t |  128
        conv |   19m |     1t |    (128, 8, 8) |    (128, 8, 8) |    9m |   9m |      1t |  128
        full |   50m |    25m |           3072 |           3072 |   25m |  25m |     25m |    3t
        full |   19m |     9m |           3072 |           3072 |    9m |   9m |      9m |    3t
        ----------------------------------------
        total   401m      35m           378368

        6.68 mins (+/- 1 order of magnitude) per epoch estimated for ec2 g2
        264.1 mb estimated for model
        """


class Experiment064(object):
    """
    comments:
    (1) no pooling on last layer
    (0) slightly bigger images
    (1) more patience
    (3) bigger hidden layers

    like 63 except:
    (1) no pooling on last layer

    results:
    pooling on last layer makes surprisingly little difference...
    """
    def __init__(p):
        p.name = 'e064'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 237
        p.epochs = 10000
        p.final_epochs = 20
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (70, 70)

        p.rng_seed = 13579

        p.learning_rule = nn.Momentum
        p.learning_rate = .4
        p.min_learning_rate = 0.0003
        p.patience = 20
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.90
        p.activation = nn.relu

        p.batch_normalize = True

        p.l1_reg = 0.000
        p.l2_reg = 0.00005
        p.dropout_conv = 0.35
        p.dropout_hidd = 0.55

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360

    def build_net(p, n):
        n.add_convolution(64, (7, 7), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(96, (5, 5), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(96, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_conv)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_hidd)
        n.add_logistic()

        """
        type | calcs | params |    neurons     |     output     | mults | adds | weights | biases
        ----------------------------------------------------------------------
        conv |   26m |     3t |   (64, 64, 64) |   (64, 32, 32) |   13m |  13m |      3t |   64
        conv |  241m |     2t |   (96, 28, 28) |   (96, 14, 14) |  120m | 120m |      2t |   96
        conv |   24m |   960  |   (96, 12, 12) |   (96, 12, 12) |   12m |  12m |    864  |   96
        conv |   22m |     1t |  (128, 10, 10) |  (128, 10, 10) |   11m |  11m |      1t |  128
        conv |   19m |     1t |    (128, 8, 8) |    (128, 8, 8) |    9m |   9m |      1t |  128
        full |   50m |    25m |           3072 |           3072 |   25m |  25m |     25m |    3t
        full |   19m |     9m |           3072 |           3072 |    9m |   9m |      9m |    3t
        ----------------------------------------
        total   401m      35m           378368

        6.68 mins (+/- 1 order of magnitude) per epoch estimated for ec2 g2
        264.1 mb estimated for model
        """


class Experiment063(object):
    """
    comments: like 57 except:
    (0) slightly bigger images
    (1) more patience
    (3) bigger hidden layers

    compared to 48 (previous high score):
    (1) batch size chosen to fit evenly into training set so there are not unused
    observations (256 -> 237)
    (2) image shape increased (64x64 -> 70x70)
    (3) patience slightly decreased (25 -> 20) but this might have coincided
    with change that made validation improvements reset frustration
    (4) half as much l2 regularization (0.0001 -> 0.00005)
    (5) significantly more dropout (0.2 -> 0.35, and 0.3 -> 0.55)
    (6) two extra convolution layers (3-5)
    (7) more filters on convolution layers (32, 64, 128 - > 64, 96, 96, 128, 128)
    (8) bigger filters on earlier layers and smaller on later layers (5, 5, 5 -> 7, 5, 3, 3, 3)
    (9) bigger hidden layers (1024 -> 3072)

    variations on 063 that don't seem to be helping so far:
    e064: no pooling on last layer (best validation 0.811 -> 0.836)
    e065: no pooling on last layer, higher dropout
    e066: no pooling on last layer, higher dropout, bigger image, add conv layer

    next try some combination of:
    (1) slightly bigger image
    (2) even more filters
    (3) another hidden layer
    (4) even more conv layers

    results:
    New personal high score!!  Quite a significant improvement too.  Probably
    a combination of increased patience (wringing more out the models) and
    bigger hidden layers/images.
    """
    def __init__(p):
        p.name = 'e063'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 237
        p.epochs = 10000
        p.final_epochs = 40
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (70, 70)

        p.rng_seed = 13579

        p.learning_rule = nn.Momentum
        p.learning_rate = .4
        p.min_learning_rate = 0.0003
        p.patience = 20
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.90
        p.activation = nn.relu

        p.batch_normalize = True

        p.l1_reg = 0.000
        p.l2_reg = 0.00005
        p.dropout_conv = 0.35
        p.dropout_hidd = 0.55

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360

    def build_net(p, n):
        n.add_convolution(64, (7, 7), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(96, (5, 5), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(96, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (3, 3), (2, 2), batch_normalize=p.batch_normalize)
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_conv)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(3072, batch_normalize=p.batch_normalize)
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_hidd)
        n.add_logistic()

        # type | calcs | params |    neurons     |     output     | mults | adds | weights | biases
        # ----------------------------------------------------------------------
        # conv |   26m |     3t |   (64, 64, 64) |   (64, 32, 32) |   13m |  13m |      3t |   64
        # conv |  241m |     2t |   (96, 28, 28) |   (96, 14, 14) |  120m | 120m |      2t |   96
        # conv |   24m |   960  |   (96, 12, 12) |   (96, 12, 12) |   12m |  12m |    864  |   96
        # conv |   22m |     1t |  (128, 10, 10) |  (128, 10, 10) |   11m |  11m |      1t |  128
        # conv |   19m |     1t |    (128, 8, 8) |    (128, 4, 4) |    9m |   9m |      1t |  128
        # full |   13m |     6m |           3072 |           3072 |    6m |   6m |      6m |    3t
        # full |   19m |     9m |           3072 |           3072 |    9m |   9m |      9m |    3t
        # ----------------------------------------
        # total   363m      16m           378368
        #
        # 6.05 mins (+/- 1 order of magnitude) per epoch estimated for ec2 g2
        # 120.1 mb estimated for model


class Experiment062(object):
    """
    comments:
    (0) 94 x 94 images
    (1) relu
    (2) batch normalization
    (3) 5 conv layers
    (4) big 7x7 filters on first conv layer
    (5) no pooling on conv layer 3 and 4
    (6) slightly higher dropout
    (7) more filters on first and second layer compared to e058
    (8) larger hidden layers compared to e057

    results:
    (initial e057 used pool strides which are broken, so I change it to this)

    Looks like regularization is really strong.  Try running with less dropout?
    """
    def __init__(p):
        p.name = 'e062'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 237
        p.epochs = 10000
        p.final_epochs = 40
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (94, 94)

        p.rng_seed = 13579

        p.learning_rule = nn.Momentum
        p.learning_rate = .4
        p.min_learning_rate = 0.00003
        p.patience = 15
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.90
        p.activation = nn.relu

        p.batch_normalize = True

        p.l1_reg = 0.000
        p.l2_reg = 0.00005
        p.dropout_conv = 0.35
        p.dropout_hidd = 0.55

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360

    def build_net(p, n):
        n.add_convolution(64, (7, 7), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(96, (5, 5), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(96, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (3, 3), (2, 2), batch_normalize=p.batch_normalize)
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_conv)
        n.add_hidden(2048, batch_normalize=p.batch_normalize)
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(2048, batch_normalize=p.batch_normalize)
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_hidd)
        n.add_logistic()

        # size estimate:
        # type | calcs | params |    neurons     |     output     | mults | adds | weights | biases
        # ----------------------------------------------------------------------
        # conv |   49m |     3t |   (64, 88, 88) |   (64, 44, 44) |   24m |  25m |      3t |   64
        # conv |  492m |     2t |   (96, 40, 40) |   (96, 20, 20) |  246m | 246m |      2t |   96
        # conv |   54m |   960  |   (96, 18, 18) |   (96, 18, 18) |   27m |  27m |    864  |   96
        # conv |   57m |     1t |  (128, 16, 16) |  (128, 16, 16) |   28m |  28m |      1t |  128
        # conv |   58m |     1t |  (128, 14, 14) |    (128, 7, 7) |   29m |  29m |      1t |  128
        # full |   26m |    13m |           2048 |           2048 |   13m |  13m |     13m |    2t
        # full |    8m |     4m |           2048 |           2048 |    4m |   4m |      4m |    2t
        # ----------------------------------------
        # total   743m      17m           742272
        #
        # 12.38 mins (+/- 1 order of magnitude) per epoch estimated for ec2 g2
        # 130.1 mb estimated for model


class Experiment061(object):
    """
    comments:
    running 4 experiments based on e051 to see if prelu ever helps
    e052 - prelu w/o  batch normalization
    e053 - relu  w/o  batch normalization
    e054 - prelu w/ batch normalization
    e055 - relu  w/ batch normalization
    e060 - prelu w/o batch normalization, lower dropout
    e061 - relu w/ batch normalization, lower dropout

    (w/o batch normalization pair have lower learning rate and smaller batch
    sizes)

    all of these have updated annealing where learning rate decay is delayed
    if either training or validation score is improving.  I think this makes
    some sense where the training data is transformed each epoch because even
    if training score is not improving the network is still possibly being
    regularized by the transformed training data, so something useful is
    potentially happening (as measured by improvement in validation score).

    results:
    """
    def __init__(p):
        p.name = 'e061'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 237
        p.epochs = 10000
        p.final_epochs = 15
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (64, 64)

        p.rng_seed = 13579

        p.learning_rule = nn.Momentum
        p.learning_rate = .4
        p.min_learning_rate = 0.0003
        p.patience = 15
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.90
        p.activation = nn.relu

        p.batch_normalize = True

        p.l1_reg = 0.000
        p.l2_reg = 0.000
        p.dropout_conv = 0.2
        p.dropout_hidd = 0.3

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360

    def build_net(p, n):
        n.add_convolution(32, (5, 5), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(64, (5, 5), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (5, 5), (2, 2), batch_normalize=p.batch_normalize)
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_conv)
        n.add_hidden(1024, batch_normalize=p.batch_normalize)
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(1024, batch_normalize=p.batch_normalize)
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_hidd)
        n.add_logistic()


class Experiment060(object):
    """
    comments:
    running 4 experiments based on e051 to see if prelu ever helps
    e052 - prelu w/o  batch normalization
    e053 - relu  w/o  batch normalization
    e054 - prelu w/ batch normalization
    e055 - relu  w/ batch normalization
    e060 - prelu w/o batch normalization, lower dropout
    e061 - relu w/ batch normalization, lower dropout

    (w/o batch normalization pair have lower learning rate and smaller batch
    sizes)

    all of these have updated annealing where learning rate decay is delayed
    if either training or validation score is improving.  I think this makes
    some sense where the training data is transformed each epoch because even
    if training score is not improving the network is still possibly being
    regularized by the transformed training data, so something useful is
    potentially happening (as measured by improvement in validation score).

    results:
    """
    def __init__(p):
        p.name = 'e060'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 237
        p.epochs = 10000
        p.final_epochs = 15
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (64, 64)

        p.rng_seed = 13579

        p.learning_rule = nn.Momentum
        p.learning_rate = .4
        p.min_learning_rate = 0.0003
        p.patience = 15
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.90
        p.activation = nn.PReLULayer

        p.batch_normalize = True

        p.l1_reg = 0.000
        p.l2_reg = 0.000
        p.dropout_conv = 0.2
        p.dropout_hidd = 0.3

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360

    def build_net(p, n):
        n.add_convolution(32, (5, 5), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(64, (5, 5), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (5, 5), (2, 2), batch_normalize=p.batch_normalize)
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_conv)
        n.add_hidden(1024, batch_normalize=p.batch_normalize)
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(1024, batch_normalize=p.batch_normalize)
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_hidd)
        n.add_logistic()


class Experiment059(object):
    """
    comments:
    (1) relu
    (2) batch normalization
    (3) 5 conv layers
    (4) no pooling on 1st and 4th
    (5) layers closer to images have large filters
    (5) slightly higher dropout

    results:
    (pool strides don't work so I revised this experiment)

     140   | 28.6%, 0.928103 | 32.4%, 1.035030 * | 29.2%, 0.956871 | learning rate=0.000400,

    Looks like regularization is really strong.  Try running with less dropout?
    """
    def __init__(p):
        p.name = 'e059'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 237
        p.epochs = 10000
        p.final_epochs = 40
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (64, 64)

        p.rng_seed = 13579

        p.learning_rule = nn.Momentum
        p.learning_rate = .4
        p.min_learning_rate = 0.0003
        p.patience = 10
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.90
        p.activation = nn.relu

        p.batch_normalize = True

        p.l1_reg = 0.000
        p.l2_reg = 0.00005
        p.dropout_conv = 0.35
        p.dropout_hidd = 0.55

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360

    def build_net(p, n):
        # n.add_convolution(32, (5, 5), (2, 2), (1, 1), batch_normalize=p.batch_normalize)
        # n.add_dropout(p.dropout_conv)
        # n.add_convolution(64, (5, 5), (3, 3), (2, 2), batch_normalize=p.batch_normalize)
        # n.add_dropout(p.dropout_conv)
        # n.add_convolution(64, (3, 3), (3, 3), (2, 2), batch_normalize=p.batch_normalize)
        # n.add_dropout(p.dropout_conv)
        # n.add_convolution(96, (3, 3), (1, 1), (1, 1), batch_normalize=p.batch_normalize)
        # n.add_dropout(p.dropout_conv)
        # n.add_convolution(64, (3, 3), (3, 3), (2, 2), batch_normalize=p.batch_normalize)

        n.add_convolution(32, (5, 5), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(64, (5, 5), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(64, (3, 3), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(96, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(64, (3, 3), (2, 2), batch_normalize=p.batch_normalize)

        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_conv)
        n.add_hidden(1280, batch_normalize=p.batch_normalize)
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(1280, batch_normalize=p.batch_normalize)
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_hidd)
        n.add_logistic()



class Experiment058(object):
    """
    comments:
    (1) relu
    (2) batch normalization
    (3) 5 conv layers
    (4) big 7x7 filters on first conv layer
    (5) no pooling on conv layer 3 and 4
    (6) slightly higher dropout
    (7) fewer filters on first and second layer compared to e057
    (8) smaller hidden layers compared to e057

    results:
    (pool strides are broken so I revised this experiment)

    152   | 27.6%, 0.890653 | 30.9%, 0.977722 * | 28.4%, 0.911837 | learning rate=0.000400,

    Looks like regularization is really strong.  Try running with less dropout?
    """
    def __init__(p):
        p.name = 'e058'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 237
        p.epochs = 10000
        p.final_epochs = 40
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (64, 64)

        p.rng_seed = 13579

        p.learning_rule = nn.Momentum
        p.learning_rate = .4
        p.min_learning_rate = 0.0003
        p.patience = 10
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.90
        p.activation = nn.relu

        p.batch_normalize = True

        p.l1_reg = 0.000
        p.l2_reg = 0.00005
        p.dropout_conv = 0.35
        p.dropout_hidd = 0.55

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360

    def build_net(p, n):
        # pool strides don't work:
        # n.add_convolution(32, (5, 5), (2, 2), (1, 1), batch_normalize=p.batch_normalize)
        # n.add_dropout(p.dropout_conv)
        # n.add_convolution(64, (5, 5), (3, 3), (2, 2), batch_normalize=p.batch_normalize)
        # n.add_dropout(p.dropout_conv)
        # n.add_convolution(64, (3, 3), (1, 1), (1, 1), batch_normalize=p.batch_normalize)
        # n.add_dropout(p.dropout_conv)
        # n.add_convolution(96, (3, 3), (1, 1), (1, 1), batch_normalize=p.batch_normalize)
        # n.add_dropout(p.dropout_conv)
        # n.add_convolution(64, (3, 3), (3, 3), (2, 2), batch_normalize=p.batch_normalize)

        # ran out of memory:
        # n.add_convolution(32, (5, 5), (1, 1), batch_normalize=p.batch_normalize)
        # n.add_dropout(p.dropout_conv)
        # n.add_convolution(64, (5, 5), (2, 2), batch_normalize=p.batch_normalize)
        # n.add_dropout(p.dropout_conv)
        # n.add_convolution(64, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        # n.add_dropout(p.dropout_conv)
        # n.add_convolution(96, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        # n.add_dropout(p.dropout_conv)
        # n.add_convolution(64, (3, 3), (2, 2), batch_normalize=p.batch_normalize)

        n.add_convolution(32, (7, 7), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(64, (5, 5), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(96, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (3, 3), (2, 2), batch_normalize=p.batch_normalize)

        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_conv)
        n.add_hidden(1280, batch_normalize=p.batch_normalize)
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(1280, batch_normalize=p.batch_normalize)
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_hidd)
        n.add_logistic()


class Experiment057(object):
    """
    comments:
    (1) relu
    (2) batch normalization
    (3) 5 conv layers
    (4) big 7x7 filters on first conv layer
    (5) no pooling on conv layer 3 and 4
    (6) slightly higher dropout
    (7) more filters on first and second layer compared to e058
    (8) larger hidden layers compared to e058

    results:
    (initial e057 used pool strides which are broken, so I change it to this)

     157   | 26.4%, 0.857902 | 28.4%, 0.876298 * | 26.9%, 0.890262 | frustration=6,

    Looks like regularization is really strong.  Try running with less dropout?
    """
    def __init__(p):
        p.name = 'e057'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 237
        p.epochs = 10000
        p.final_epochs = 40
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (64, 64)

        p.rng_seed = 13579

        p.learning_rule = nn.Momentum
        p.learning_rate = .4
        p.min_learning_rate = 0.0003
        p.patience = 10
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.90
        p.activation = nn.relu

        p.batch_normalize = True

        p.l1_reg = 0.000
        p.l2_reg = 0.00005
        p.dropout_conv = 0.35
        p.dropout_hidd = 0.55

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360

    def build_net(p, n):
        # pool strides don't work:
        # n.add_convolution(32, (5, 5), (2, 2), (1, 1), batch_normalize=p.batch_normalize)
        # n.add_dropout(p.dropout_conv)
        # n.add_convolution(64, (5, 5), (3, 3), (2, 2), batch_normalize=p.batch_normalize)
        # n.add_dropout(p.dropout_conv)
        # n.add_convolution(96, (3, 3), (1, 1), (1, 1), batch_normalize=p.batch_normalize)
        # n.add_dropout(p.dropout_conv)
        # n.add_convolution(96, (3, 3), (1, 1), (1, 1), batch_normalize=p.batch_normalize)
        # n.add_dropout(p.dropout_conv)
        # n.add_convolution(128, (3, 3), (3, 3), (2, 2), batch_normalize=p.batch_normalize)

        n.add_convolution(64, (7, 7), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(96, (5, 5), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(96, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (3, 3), (2, 2), batch_normalize=p.batch_normalize)
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_conv)
        n.add_hidden(2048, batch_normalize=p.batch_normalize)
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(2048, batch_normalize=p.batch_normalize)
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_hidd)
        n.add_logistic()

        # size estimate:
        # type | calcs | params |     output     | mults | adds | weights | biases
        # ----------------------------------------------------------------------
        # conv |   21m |     3t |   (64, 58, 58) |   11m |  11m |      3t |   64
        # conv |  192m |     2t |   (96, 25, 25) |   96m |  96m |      2t |   96
        # conv |   17m |   960  |   (96, 10, 10) |    8m |   8m |    864  |   96
        # conv |   14m |     1t |    (128, 8, 8) |    7m |   7m |      1t |  128
        # conv |   11m |     1t |    (128, 6, 6) |    5m |   5m |      1t |  128
        # full |    5m |     2m |           2048 |    2m |   2m |      2m |    2t
        # full |    8m |     4m |           2048 |    4m |   4m |      4m |    2t
        # ----------------------------------------
        # total   268m       7m           301792

        # 4.46 mins (+/- 1 order of magnitude) per epoch estimated for ec2 g2
        # 50.1 mb estimated for model

        # ('conv', {'filter': (64, 7, 7), 'pool': (2, 2), 'pool_stride': (2, 2)}),
        # ('conv', {'filter': (96, 5, 5), 'pool': (2, 2), 'pool_stride': (2, 2)}),
        # ('conv', {'filter': (96, 3, 3), 'pool': (1, 1), 'pool_stride': (1, 1)}),
        # ('conv', {'filter': (128, 3, 3), 'pool': (1, 1), 'pool_stride': (1, 1)}),
        # ('conv', {'filter': (128, 3, 3), 'pool': (2, 2), 'pool_stride': (2, 2)}),
        # ('full', {'num': 2048}),
        # ('full', {'num': 2048})


class Experiment056(object):
    """
    comments:
    (1) relu
    (2) batch normalization
    (3) 4 conv layers
    (4) no pooling on first and last conv layer
    (5) slightly higher dropout

    results:
    This seems to have done ok.  But compared to e048 (my current high scorer)
    didn't end up with as low score, and trained slower... but it was more
    highly regularized and the training score never diverged from the validation
    score, whereas with e048 it diverged pretty strongly.

    Maybe I should rerun this with a little less dropout... or other deep
    models anyway...

    000   | 64.5%, 2.403353 | 74.6%, 3.079786   | 64.0%, 2.413033 | train time: 317.2, test time: 10.7, learning rate=0.400000,
    093   | 26.5%, 0.854883 | 27.3%, 0.856575 * | 27.3%, 0.871424 | momentum=0.856, frustration=3,
    """
    def __init__(p):
        p.name = 'e056'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 237
        p.epochs = 10000
        p.final_epochs = 40
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (64, 64)

        p.rng_seed = 13579

        p.learning_rule = nn.Momentum
        p.learning_rate = .4
        p.min_learning_rate = 0.0003
        p.patience = 10
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.90
        p.activation = nn.relu

        p.batch_normalize = True

        p.l1_reg = 0.000
        p.l2_reg = 0.00005
        p.dropout_conv = 0.35
        p.dropout_hidd = 0.55

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360

    def build_net(p, n):
        n.add_convolution(32, (3, 3), (1, 1), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(64, (3, 3), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(96, (5, 5), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (5, 5), (1, 1), batch_normalize=p.batch_normalize)
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_conv)
        n.add_hidden(1280, batch_normalize=p.batch_normalize)
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(1280, batch_normalize=p.batch_normalize)
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_hidd)
        n.add_logistic()


class Experiment055(object):
    """
    comments:
    running 4 experiments based on e051 to see if prelu ever helps
    e052 - prelu w/o  batch normalization
    e053 - relu  w/o  batch normalization
    e054 - prelu w/ batch normalization
    e055 - relu  w/ batch normalization

    (w/o batch normalization pair has lower learning rate and smaller batch
    sizes)

    all of these have updated annealing where learning rate decay is delayed
    if either training or validation score is improving.  I think this makes
    some sense where the training data is transformed each epoch because even
    if training score is not improving the network is still possibly being
    regularized by the transformed training data, so something useful is
    potentially happening (as measured by improvement in validation score).

    results:
    """
    def __init__(p):
        p.name = 'e055'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 237
        p.epochs = 10000
        p.final_epochs = 40
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (64, 64)

        p.rng_seed = 13579

        p.learning_rule = nn.Momentum
        p.learning_rate = .4
        p.min_learning_rate = 0.0003
        p.patience = 10
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.90
        p.activation = nn.relu

        p.batch_normalize = True

        p.l1_reg = 0.000
        p.l2_reg = 0.000
        p.dropout_conv = 0.3
        p.dropout_hidd = 0.5

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360

    def build_net(p, n):
        n.add_convolution(32, (5, 5), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(64, (5, 5), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (5, 5), (2, 2), batch_normalize=p.batch_normalize)
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_conv)
        n.add_hidden(1024, batch_normalize=p.batch_normalize)
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(1024, batch_normalize=p.batch_normalize)
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_hidd)
        n.add_logistic()


class Experiment054(object):
    """
    comments:
    running 4 experiments based on e051 to see if prelu ever helps
    e052 - prelu w/o  batch normalization
    e053 - relu  w/o  batch normalization
    e054 - prelu w/ batch normalization
    e055 - relu  w/ batch normalization

    (w/o batch normalization pair have lower learning rate and smaller batch
    sizes)

    all of these have updated annealing where learning rate decay is delayed
    if either training or validation score is improving.  I think this makes
    some sense where the training data is transformed each epoch because even
    if training score is not improving the network is still possibly being
    regularized by the transformed training data, so something useful is
    potentially happening (as measured by improvement in validation score).

    results:
    """
    def __init__(p):
        p.name = 'e054'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 237
        p.epochs = 10000
        p.final_epochs = 40
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (64, 64)

        p.rng_seed = 13579

        p.learning_rule = nn.Momentum
        p.learning_rate = .4
        p.min_learning_rate = 0.0003
        p.patience = 10
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.90
        p.activation = nn.PReLULayer

        p.batch_normalize = True

        p.l1_reg = 0.000
        p.l2_reg = 0.000
        p.dropout_conv = 0.3
        p.dropout_hidd = 0.5

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360

    def build_net(p, n):
        n.add_convolution(32, (5, 5), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(64, (5, 5), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (5, 5), (2, 2), batch_normalize=p.batch_normalize)
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_conv)
        n.add_hidden(1024, batch_normalize=p.batch_normalize)
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(1024, batch_normalize=p.batch_normalize)
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_hidd)
        n.add_logistic()


class Experiment053(object):
    """
    comments:
    running 4 experiments based on e051 to see if prelu ever helps
    e052 - prelu w/o  batch normalization
    e053 - relu  w/o  batch normalization
    e054 - prelu w/ batch normalization
    e055 - relu  w/ batch normalization

    (w/o batch normalization pair have lower learning rate and smaller batch
    sizes)

    all of these have updated annealing where learning rate decay is delayed
    if either training or validation score is improving.  I think this makes
    some sense where the training data is transformed each epoch because even
    if training score is not improving the network is still possibly being
    regularized by the transformed training data, so something useful is
    potentially happening (as measured by improvement in validation score).

    results:
    """
    def __init__(p):
        p.name = 'e053'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 64
        p.epochs = 10000
        p.final_epochs = 40
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (64, 64)

        p.rng_seed = 13579

        p.learning_rule = nn.Momentum
        p.learning_rate = .08
        p.min_learning_rate = 0.0003
        p.patience = 10
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.90
        p.activation = nn.relu

        p.batch_normalize = False

        p.l1_reg = 0.000
        p.l2_reg = 0.000
        p.dropout_conv = 0.3
        p.dropout_hidd = 0.5

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360

    def build_net(p, n):
        n.add_convolution(32, (5, 5), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(64, (5, 5), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (5, 5), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.merge_data_channel('shapes')
        n.add_hidden(1024, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.merge_data_channel('shapes')
        n.add_hidden(1024, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.merge_data_channel('shapes')
        n.add_logistic()


class Experiment052(object):
    """
    comments:
    running 4 experiments based on e051 to see if prelu ever helps
    e052 - prelu w/o  batch normalization
    e053 - relu  w/o  batch normalization
    e054 - prelu w/ batch normalization
    e055 - relu  w/ batch normalization

    (w/o batch normalization pair have lower learning rate and smaller batch
    sizes)

    all of these have updated annealing where learning rate decay is delayed
    if either training or validation score is improving.  I think this makes
    some sense where the training data is transformed each epoch because even
    if training score is not improving the network is still possibly being
    regularized by the transformed training data, so something useful is
    potentially happening (as measured by improvement in validation score).

    results:
    In both cases (w/ and w/o bn) prelu did substantially worse than relu
    in both validation and training score.  However, I did notice something
    interesting.  Prelu validation score tended to be substantially better
    than training score, whereas relu validation score tended to be about the
    same or a little worse.  If prelu is having some regularization effect,
    perhaps the worse scores are merely a symptom of that.  I wonder what
    would happen if I turned down dropout a little.
    """
    def __init__(p):
        p.name = 'e052'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 64
        p.epochs = 10000
        p.final_epochs = 40
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (64, 64)

        p.rng_seed = 13579

        p.learning_rule = nn.Momentum
        p.learning_rate = .08
        p.min_learning_rate = 0.0003
        p.patience = 10
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.90
        p.activation = nn.PReLULayer

        p.batch_normalize = False

        p.l1_reg = 0.000
        p.l2_reg = 0.000
        p.dropout_conv = 0.3
        p.dropout_hidd = 0.5

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360

    def build_net(p, n):
        n.add_convolution(32, (5, 5), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(64, (5, 5), (2, 2), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (5, 5), (2, 2), batch_normalize=p.batch_normalize)
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_conv)
        n.add_hidden(1024, batch_normalize=p.batch_normalize)
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(1024, batch_normalize=p.batch_normalize)
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_hidd)
        n.add_logistic()



class Experiment051(object):
    """
    comments:  same as e037 except:
    (1) batch normalization
    (2) prelu (recommended not to use with l1/l2 reg but last experiment did)
    (3) higher learning rate
    (4) shape input comes before dropout
    (5) lower patience (given that rotations are regularizing even when training
        is not improving this might be a bad idea)
    (6) bigger batch sizes

    results:
    Seems to have done very poorly.  I think the annealing rate may have been
    too aggressive.  And this version of batch normalization had bug on
    change of learning rate that might have prevented it improved validation
    scores for a while after learning rate change.

    Not sure I can draw any conclusions from this experiment except to note
    that it's another example of prelu not doing particularly well, but
    being very well regularized.

    091   | 27.7%, 0.902754 | 30.8%, 0.967690 * | 28.4%, 0.925873 | momentum=0.752, frustration=9,
    """
    def __init__(p):
        p.name = 'e051'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 237
        p.epochs = 10000
        p.final_epochs = 40
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (64, 64)

        p.rng_seed = 13579

        p.learning_rule = nn.Momentum
        p.learning_rate = .4
        p.min_learning_rate = 0.0003
        p.patience = 10
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.90
        p.activation = nn.PReLULayer

        p.l1_reg = 0.000
        p.l2_reg = 0.000
        p.dropout_conv = 0.3
        p.dropout_hidd = 0.5

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360

    def build_net(p, n):
        n.add_convolution(32, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(64, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (5, 5), (2, 2))
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_conv)
        n.add_hidden(1024)
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(1024)
        n.merge_data_channel('shapes')
        n.add_dropout(p.dropout_hidd)
        n.add_logistic()


class Experiment050(object):
    """
    comments:
    (1) prelu!

    otherwise the same as e048:
    (0) new ema inference statistics for batch normalization layer
    (1) light dropout
    (2) light l2 regularization
    (3) large batches (to increase sample size)
    (4) 0.4 learning rate

    results:
    looks slightly worse than relu.  BUT, I just noticed that they say
    not to use weight decay with prelu because it biases the parameter to be
    small making it very similar to relu.  Run another experiment without
    l2 regularization.

    Interesting!  In subsequent experiments I've noticed that prelu seems to
    regularize a bit.  Here it does NOT seem to be doing that.  But here there
    is l2 regulariation which biases prelu towards relu.  Maybe this is further
    evidence that prelu is working as a regularizer?

    Interesting.  This is almost a high score:
    113   | 26.1%, 0.838854 | 22.3%, 0.670970 * | 26.2%, 0.847457 | frustration=4,

    compared to e048 (my current high score):
    113   | 25.4%, 0.836218 | 22.4%, 0.673820   | 26.9%, 0.850391 | frustration=1,

    But this (e050) didn't have a chance to improve at lower training rates...
    whereas e048 did.  Both networks have very similar architecture so I suppose
    it's not surprising that they have similar scores.
    """
    def __init__(p):
        p.name = 'e050'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 256
        p.epochs = 10000
        p.final_epochs = 40
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (64, 64)

        p.rng_seed = 13579

        p.learning_rule = nn.Momentum
        p.learning_rate = .4
        p.min_learning_rate = 0.0003
        p.patience = 25
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.90
        p.activation = nn.PReLULayer

        p.l1_reg = 0.000
        p.l2_reg = 0.0001
        p.dropout_conv = 0.2
        p.dropout_hidd = 0.3

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360

    def build_net(p, n):
        n.add_convolution(32, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(64, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.merge_data_channel('shapes')
        n.add_hidden(1024)
        n.add_dropout(p.dropout_hidd)
        n.merge_data_channel('shapes')
        n.add_hidden(1024)
        n.add_dropout(p.dropout_hidd)
        n.merge_data_channel('shapes')
        n.add_logistic()


class Experiment049(object):
    """
    comments:
    (0) new ema inference statistics for batch normalization layer
    (1) no dropout
    (2) stronger l2 regularization
    (3) large batches (to increase sample size)
    (4) 0.4 learning rate

    I ran a similar experiment with the same l2 regularization as e048 for
    comparison, but was accidentally killed without writing output.  It was
    not doing particularly well, so I decided instead of rerunning it to
    try again with higher l2 regularization.

    results:
    I thought I was going to break my high score with a best validation
    score of 0.849262, but Kaggle considered my entry a 0.957990.  That is
    such a huge discrepancy that it seems like it must be due to the final
    training.  Perhaps without the dropout the final training caused extreme
    overfitting instead of improving with the extra training examples.  A little
    disapointing not to see the leaderboard improvement, but not worth
    fiddling with since I'm pretty sure I can beat this score with future experiments.
    Particularly, seems to do better with dropout despite what the
    batch normalization paper says.
    """
    def __init__(p):
        p.name = 'e049'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 256
        p.epochs = 10000
        p.final_epochs = 40
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (64, 64)

        p.rng_seed = 13579

        p.learning_rule = nn.Momentum
        p.learning_rate = .4
        p.min_learning_rate = 0.0003
        p.patience = 25
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.90
        p.activation = nn.relu

        p.l1_reg = 0.000
        p.l2_reg = 0.0005

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360

    def build_net(p, n):
        n.add_convolution(32, (5, 5), (2, 2))
        n.add_convolution(64, (5, 5), (2, 2))
        n.add_convolution(128, (5, 5), (2, 2))
        n.merge_data_channel('shapes')
        n.add_hidden(1024)
        n.merge_data_channel('shapes')
        n.add_hidden(1024)
        n.merge_data_channel('shapes')
        n.add_logistic()


class Experiment048(object):
    """
    comments:
    (0) new ema inference statistics for batch normalization layer
    (1) light dropout
    (2) light l2 regularization
    (3) large batches (to increase sample size)
    (4) 0.4 learning rate

    results:
    Hi score!!!
    """
    def __init__(p):
        p.name = 'e048'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 256
        p.epochs = 10000
        p.final_epochs = 40
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (64, 64)

        p.rng_seed = 13579

        p.learning_rule = nn.Momentum
        p.learning_rate = .4
        p.min_learning_rate = 0.0003
        p.patience = 25
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.90
        p.activation = nn.relu

        p.l1_reg = 0.000
        p.l2_reg = 0.0001
        p.dropout_conv = 0.2
        p.dropout_hidd = 0.3

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360

    def build_net(p, n):
        n.add_convolution(32, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(64, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.merge_data_channel('shapes')
        n.add_hidden(1024)
        n.add_dropout(p.dropout_hidd)
        n.merge_data_channel('shapes')
        n.add_hidden(1024)
        n.add_dropout(p.dropout_hidd)
        n.merge_data_channel('shapes')
        n.add_logistic()


class Experiment047(object):
    """
    comments: same as e039 except:
    (1) higher learning rate
    (2) no dropout
    (3) new population statistics implementation for batch normalization
    (4) l2 regularization

    same as e041 except:
    (1) new population statistics implementation for batch normalization
    (2) l2 regularization

    same as e042 except
    (1) l2 regularization

    same as 044 except
    (1) *higher* l2 regularization

    results:
    looked like it was doing well initially and then started doing
    horribly... strongly diverging from train loss.

    """
    def __init__(p):
        p.name = 'e047'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 256
        p.epochs = 10000
        p.final_epochs = 40
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (64, 64)

        p.rng_seed = 13579

        p.learning_rule = nn.Momentum
        p.learning_rate = .36
        p.min_learning_rate = 0.0003
        p.patience = 25
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.90
        p.activation = nn.relu

        p.l1_reg = 0.000
        p.l2_reg = 0.0005
        p.dropout_conv = 0.3
        p.dropout_hidd = 0.5

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360

    def build_net(p, n):
        n.add_convolution(32, (5, 5), (2, 2))
        n.add_convolution(64, (5, 5), (2, 2))
        n.add_convolution(128, (5, 5), (2, 2))
        n.merge_data_channel('shapes')
        n.add_hidden(1024)
        n.merge_data_channel('shapes')
        n.add_hidden(1024)
        n.merge_data_channel('shapes')
        n.add_logistic()


class Experiment046(object):
    """
    comments:
    same as e043 except:
    (1) smaller batches
    (2) higher dropout

    same as e045 except:
    (1) higher dropout

    results:
    """
    def __init__(p):
        p.name = 'e046'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 64
        p.epochs = 10000
        p.final_epochs = 40
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (64, 64)

        p.rng_seed = 13579

        p.learning_rule = nn.Momentum
        p.learning_rate = .36
        p.min_learning_rate = 0.0003
        p.patience = 25
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.90
        p.activation = nn.relu

        p.l1_reg = 0.000
        p.l2_reg = 0.000
        p.dropout_conv = 0.5
        p.dropout_hidd = 0.6

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360

    def build_net(p, n):
        n.add_convolution(32, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(64, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.merge_data_channel('shapes')
        n.add_hidden(1024)
        n.add_dropout(p.dropout_hidd)
        n.merge_data_channel('shapes')
        n.add_hidden(1024)
        n.add_dropout(p.dropout_hidd)
        n.merge_data_channel('shapes')
        n.add_logistic()


class Experiment045(object):
    """
    comments:
    same as e043 except:
    (1) smaller batches

    same as e039 except:
    (1) higher learning rate
    (2) smaller batches
    (3) new population statistics implementation for batch normalization

    same as e041 except:
    (1) new population statistics implementation for batch normalization
    (2) added dropout back
    (3) smaller batches

    same as e042 except:
    (1) added dropout back
    (2) smaller batches

    same as e038 except:
    (1) bigger batches
    (2) higher learning rate
    (3) new population statistics implementation for batch normalization

    results:
    """
    def __init__(p):
        p.name = 'e045'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 64
        p.epochs = 10000
        p.final_epochs = 40
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (64, 64)

        p.rng_seed = 13579

        p.learning_rule = nn.Momentum
        p.learning_rate = .36
        p.min_learning_rate = 0.0003
        p.patience = 25
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.90
        p.activation = nn.relu

        p.l1_reg = 0.000
        p.l2_reg = 0.000
        p.dropout_conv = 0.3
        p.dropout_hidd = 0.5

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360

    def build_net(p, n):
        n.add_convolution(32, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(64, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.merge_data_channel('shapes')
        n.add_hidden(1024)
        n.add_dropout(p.dropout_hidd)
        n.merge_data_channel('shapes')
        n.add_hidden(1024)
        n.add_dropout(p.dropout_hidd)
        n.merge_data_channel('shapes')
        n.add_logistic()


class Experiment044(object):
    """
    comments: same as e039 except:
    (1) higher learning rate
    (2) no dropout
    (3) new population statistics implementation for batch normalization
    (4) l2 regularization

    same as e041 except:
    (1) new population statistics implementation for batch normalization
    (2) l2 regularization

    same as e042 except
    (1) l2 regularization

    results:

    """
    def __init__(p):
        p.name = 'e044'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 256
        p.epochs = 10000
        p.final_epochs = 40
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (64, 64)

        p.rng_seed = 13579

        p.learning_rule = nn.Momentum
        p.learning_rate = .36
        p.min_learning_rate = 0.0003
        p.patience = 25
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.90
        p.activation = nn.relu

        p.l1_reg = 0.000
        p.l2_reg = 0.0001
        p.dropout_conv = 0.3
        p.dropout_hidd = 0.5

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360

    def build_net(p, n):
        n.add_convolution(32, (5, 5), (2, 2))
        n.add_convolution(64, (5, 5), (2, 2))
        n.add_convolution(128, (5, 5), (2, 2))
        n.merge_data_channel('shapes')
        n.add_hidden(1024)
        n.merge_data_channel('shapes')
        n.add_hidden(1024)
        n.merge_data_channel('shapes')
        n.add_logistic()


class Experiment043(object):
    """
    comments: same as e039 except:
    (1) higher learning rate
    (3) new population statistics implementation for batch normalization

    same as e041 except:
    (1) new population statistics implementation for batch normalization
    (2) added dropout back

    same as e042 except:
    (1) added dropout back

    same as e038 except:
    (1) bigger batches
    (2) higher learning rate
    (3) new population statistics implementation for batch normalization

    results:
    """
    def __init__(p):
        p.name = 'e043'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 256
        p.epochs = 10000
        p.final_epochs = 40
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (64, 64)

        p.rng_seed = 13579

        p.learning_rule = nn.Momentum
        p.learning_rate = .36
        p.min_learning_rate = 0.0003
        p.patience = 25
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.90
        p.activation = nn.relu

        p.l1_reg = 0.000
        p.l2_reg = 0.000
        p.dropout_conv = 0.3
        p.dropout_hidd = 0.5

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360

    def build_net(p, n):
        n.add_convolution(32, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(64, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.merge_data_channel('shapes')
        n.add_hidden(1024)
        n.add_dropout(p.dropout_hidd)
        n.merge_data_channel('shapes')
        n.add_hidden(1024)
        n.add_dropout(p.dropout_hidd)
        n.merge_data_channel('shapes')
        n.add_logistic()


class Experiment042(object):
    """
    comments: same as e039 except:
    (1) higher learning rate
    (2) no dropout
    (3) new population statistics implementation for batch normalization

    same as e041 except:
    (3) new population statistics implementation for batch normalization

    results:
    looks like it is overfitting pretty severely by epoch 5.  Maybe try a
    little bit of l2 regularization.
    """
    def __init__(p):
        p.name = 'e042'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 256
        p.epochs = 10000
        p.final_epochs = 40
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (64, 64)

        p.rng_seed = 13579

        p.learning_rule = nn.Momentum
        p.learning_rate = .36
        p.min_learning_rate = 0.0003
        p.patience = 25
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.90
        p.activation = nn.relu

        p.l1_reg = 0.000
        p.l2_reg = 0.000

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360

    def build_net(p, n):
        n.add_convolution(32, (5, 5), (2, 2))
        n.add_convolution(64, (5, 5), (2, 2))
        n.add_convolution(128, (5, 5), (2, 2))
        n.merge_data_channel('shapes')
        n.add_hidden(1024)
        n.merge_data_channel('shapes')
        n.add_hidden(1024)
        n.merge_data_channel('shapes')
        n.add_logistic()


class Experiment041(object):
    """
    comments: same as e039 except:
    (1) higher learning rate
    (2) no dropout

    results:
    """
    def __init__(p):
        p.name = 'e041'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 256
        p.epochs = 10000
        p.final_epochs = 40
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (64, 64)

        p.rng_seed = 13579

        p.learning_rule = nn.Momentum
        p.learning_rate = .36
        p.min_learning_rate = 0.0003
        p.patience = 25
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.90
        p.activation = nn.relu

        p.l1_reg = 0.000
        p.l2_reg = 0.000
        p.dropout_conv = 0.3
        p.dropout_hidd = 0.5

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360

    def build_net(p, n):
        n.add_convolution(32, (5, 5), (2, 2))
        n.add_convolution(64, (5, 5), (2, 2))
        n.add_convolution(128, (5, 5), (2, 2))
        n.merge_data_channel('shapes')
        n.add_hidden(1024)
        n.merge_data_channel('shapes')
        n.add_hidden(1024)
        n.merge_data_channel('shapes')
        n.add_logistic()


class Experiment040(object):
    """
    comments: same as e039 except:
    (1) higher learning rate

    results:
    """
    def __init__(p):
        p.name = 'e040'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 256
        p.epochs = 10000
        p.final_epochs = 40
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (64, 64)

        p.rng_seed = 13579

        p.learning_rule = nn.Momentum
        p.learning_rate = 1
        p.min_learning_rate = 0.0003
        p.patience = 25
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.90
        p.activation = nn.relu

        p.l1_reg = 0.000
        p.l2_reg = 0.000
        p.dropout_conv = 0.3
        p.dropout_hidd = 0.5

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360

    def build_net(p, n):
        n.add_convolution(32, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(64, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.merge_data_channel('shapes')
        n.add_hidden(1024)
        n.add_dropout(p.dropout_hidd)
        n.merge_data_channel('shapes')
        n.add_hidden(1024)
        n.add_dropout(p.dropout_hidd)
        n.merge_data_channel('shapes')
        n.add_logistic()


class Experiment039(object):
    """
    comments: same as e038 (which is the same as e037, my current high score
    but with batch normalization) except:
    (1) bigger batches


    results:
    """
    def __init__(p):
        p.name = 'e039'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 256
        p.epochs = 10000
        p.final_epochs = 40
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (64, 64)

        p.rng_seed = 13579

        p.learning_rule = nn.Momentum
        p.learning_rate = 0.08
        p.min_learning_rate = 0.0003
        p.patience = 25
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.90
        p.activation = nn.relu

        p.l1_reg = 0.000
        p.l2_reg = 0.000
        p.dropout_conv = 0.3
        p.dropout_hidd = 0.5

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360

    def build_net(p, n):
        n.add_convolution(32, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(64, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.merge_data_channel('shapes')
        n.add_hidden(1024)
        n.add_dropout(p.dropout_hidd)
        n.merge_data_channel('shapes')
        n.add_hidden(1024)
        n.add_dropout(p.dropout_hidd)
        n.merge_data_channel('shapes')
        n.add_logistic()


class Experiment038(object):
    """
    comments: same as e037 except:
    (1) using batch normalization (now automatically added by conv and hidden
    methods


    results:
    """
    def __init__(p):
        p.name = 'e038'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 64
        p.epochs = 10000
        p.final_epochs = 40
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (64, 64)

        p.rng_seed = 13579

        p.learning_rule = nn.Momentum
        p.learning_rate = 0.08
        p.min_learning_rate = 0.0003
        p.patience = 25
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.90
        p.activation = nn.relu

        p.l1_reg = 0.000
        p.l2_reg = 0.000
        p.dropout_conv = 0.3
        p.dropout_hidd = 0.5

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360

    def build_net(p, n):
        n.add_convolution(32, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(64, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.merge_data_channel('shapes')
        n.add_hidden(1024)
        n.add_dropout(p.dropout_hidd)
        n.merge_data_channel('shapes')
        n.add_hidden(1024)
        n.add_dropout(p.dropout_hidd)
        n.merge_data_channel('shapes')
        n.add_logistic()


class Experiment037(object):
    """
    comments:  same as e026 except:
    (1) rotator360 updated to visit each of 360 degree rotations


    results:
    """
    def __init__(p):
        p.name = 'e037'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 64
        p.epochs = 10000
        p.final_epochs = 40
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (64, 64)

        p.rng_seed = 13579

        p.learning_rule = 'momentum'
        p.learning_rate = 0.08
        p.min_learning_rate = 0.0003
        p.patience = 25
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.90
        p.activation = nn.relu

        p.l1_reg = 0.000
        p.l2_reg = 0.000
        p.dropout_conv = 0.3
        p.dropout_hidd = 0.5

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360

    def build_net(p, n):
        n.add_convolution(32, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(64, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_shape_input()
        n.add_hidden(1024)
        n.add_dropout(p.dropout_hidd)
        n.add_shape_input()
        n.add_hidden(1024)
        n.add_dropout(p.dropout_hidd)
        n.add_shape_input()
        n.add_logistic()


class Experiment036(object):
    """
    comments:  same as e026 except:
    (1) training with adadelta


    results:
    Nothing to write home about.  Seems to keep learning very slowly for a long
    time before learning slows down so much that it is time for a learning rate change.
    Learning rate decrease does seem to improve learning once it has stalled out.
    At least at a learning rate of 1, learning started off very fast and rapidly slowed
    to a crawl.  I killed it, and out of curiousity I'm going to run it again with
    learning rate 10 times lower.

    """
    def __init__(p):
        p.name = 'e036'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 64
        p.epochs = 10000
        p.final_epochs = 40
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (64, 64)

        p.rng_seed = 13579

        p.learning_rule = 'adadelta'
        p.learning_rate = 1
        p.min_learning_rate = 0.1
        p.rho = 0.95
        p.epsilon = 1e-6
        p.patience = 25
        p.improvement_threshold = 0.995

        p.activation = nn.relu

        p.l1_reg = 0.000
        p.l2_reg = 0.000
        p.dropout_conv = 0.3
        p.dropout_hidd = 0.5

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360

    def build_net(p, n):
        n.add_convolution(32, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(64, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_shape_input()
        n.add_hidden(1024)
        n.add_dropout(p.dropout_hidd)
        n.add_shape_input()
        n.add_hidden(1024)
        n.add_dropout(p.dropout_hidd)
        n.add_shape_input()
        n.add_logistic()


class Experiment035(object):
    """
    comments:  same as e026, but training on all data for a number of epochs
    equal to the epoch number upon which e026 got it's best validation score.
    My hope is that that amount of training will also be optimal for the same
    network training on a slightly larger dataset.

    Just looking for something to give me a slightly boost to raise my spirits!

    results:
    TERMINATED BY AMAZON, but it looked like it was training about as fast
    as e026 did.  (Ignore validation numbers because they are meaningless as there
    was no validation set.)

    e035:
    243   | 23.4%, 0.606748 | 27.0%, 0.819761   |                 |
    244   | 21.9%, 0.619858 | 27.4%, 0.825713 * |                 |
    245   | 20.3%, 0.645830 | 27.3%, 0.822213 * |                 |
    246   | 20.3%, 0.623338 | 27.3%, 0.817752   |                 |
    247   | 23.4%, 0.626814 | 27.4%, 0.827466 * |                 |
    248   | 21.9%, 0.626518 | 26.9%, 0.816702   |                 |
    249   | 23.4%, 0.612468 | 27.4%, 0.826046 * |                 |
    250   | 21.9%, 0.578804 | 27.6%, 0.827862 * | 21.9%, 0.578804 |
    251   | 21.9%, 0.588140 | 27.0%, 0.811296   |                 |
    252   | 23.4%, 0.617415 | 26.9%, 0.817605 * |                 |

    e026:
    251   | 27.7%, 0.905205 | 27.5%, 0.824192   |                 |
    252   | 27.8%, 0.911528 | 27.5%, 0.826194 * |                 |
    253   | 28.5%, 0.919094 | 27.6%, 0.819452   |                 |


    """
    def __init__(p):
        p.name = 'e035b'
        p.num_images = None
        p.train_pct = 100
        p.valid_pct = 0
        p.test_pct = 0
        p.num_submission_images = None

        p.batch_size = 64
        p.epochs = 318
        p.final_epochs = 0
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (64, 64)

        p.rng_seed = 13579

        p.learning_rate = 0.08
        p.min_learning_rate = 0.0003
        p.patience = 25
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.90
        p.activation = nn.relu

        p.l1_reg = 0.000
        p.l2_reg = 0.000
        p.dropout_conv = 0.3
        p.dropout_hidd = 0.5

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360

    def build_net(p, n):
        n.add_convolution(32, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(64, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_shape_input()
        n.add_hidden(1024)
        n.add_dropout(p.dropout_hidd)
        n.add_shape_input()
        n.add_hidden(1024)
        n.add_dropout(p.dropout_hidd)
        n.add_shape_input()
        n.add_logistic()


class Experiment034(object):
    """
    comments:  same as e026 except:
    (1) using Canonicalizer
    (2) using ContiguousBoxPreserveAspectRatioResizer
    (3) dropout on on data

    same as e030 except:
    (1) using ContiguousBoxPreserveAspectRatioResizer
    (2) dropout on on data

    same as e033 except:
    (1) dropout on on data


    results:
    Looks like a total failure.  Seems to be generalizing *worse* (in addition
    to learning much slower).  Compared to e033, roughly the same experiment
    but without dropout on data, the same validation score comes much later and
    corresponds to a much better training score.  So basically dropout on data
    is doing the opposite of what I expected it to do.  I'm going to try it on reversed
    images just to make sure it is not an issue of adding white vs black noise.

    e034a:
    epoch | valid err, loss | train err, loss   | test err, loss  | other | (learn rate: 0.026667)
    133   | 34.4%, 1.187444 | 30.4%, 0.943533   | 34.2%, 1.171061 |
    epoch | valid err, loss | train err, loss   | test err, loss  | other | (learn rate: 0.008889)
    181   | 36.2%, 1.279278 | 26.6%, 0.811427 * |                 |

    e033:
    031   | 34.6%, 1.186687 | 37.7%, 1.250437   | 34.4%, 1.199518 |

    e034c (on special branch where images are reversed black/white):
    so far looks roughly the same as e034a.  By epoch 86 it's starting to look
    a little better than e034a, but still much worse than e033.
    looks like it is over fitting pretty severley after epoch 173 at around 1.09 validation
    loss.
    epoch | valid err, loss | train err, loss   | test err, loss  | other | (learn rate: 0.008889)
    172   | 32.6%, 1.095105 | 28.5%, 0.862509 * | 32.3%, 1.163980 |
    199   | 33.6%, 1.138508 | 26.8%, 0.797962 * |                 |
    killing it.

    INTERESTING:  e034c (dropout with black/white reversed) seems to have done significantly
    better than e034a (dropout without black/white reversed) which makes intuitive sense
    to me for some reason.  If I try dropout on data layers in future experiments
    I should make sure to try reversing the images first.  But in any case they both
    did much worse than e033 which had no dropout on data layer... so not that exciting.

    """
    def __init__(p):
        p.name = 'e034c'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 64
        p.epochs = 10000
        p.final_epochs = 40
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (64, 64)

        p.rng_seed = 13579

        p.learning_rate = 0.08
        p.min_learning_rate = 0.0003
        p.patience = 25
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.90
        p.activation = nn.relu

        p.l1_reg = 0.000
        p.l2_reg = 0.000
        p.dropout_conv = 0.3
        p.dropout_hidd = 0.5

        p.resizer = pp.ContiguousBoxPreserveAspectRatioResizer
        # p.resizer = nn.ContiguousBoxStretchResizer
        p.contiguous_box_threshold = 2

        # p.preprocessor = nn.Rotator360
        p.preprocessor = pp.Canonicalizer

    def build_net(p, n):
        n.add_dropout(p.dropout_conv)
        n.add_convolution(32, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(64, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_shape_input()
        n.add_hidden(1024)
        n.add_dropout(p.dropout_hidd)
        n.add_shape_input()
        n.add_hidden(1024)
        n.add_dropout(p.dropout_hidd)
        n.add_shape_input()
        n.add_logistic()


class Experiment033(object):
    """
    comments:  same as e026 except:
    (1) using Canonicalizer
    (2) using ContiguousBoxPreserveAspectRatioResizer


    results:

    Extremely similar to e032 (same but with ContiguousBoxStretchResizer),
    both of which were worse than using a simple stretch resizer.

    epoch | valid err, loss | train err, loss   | test err, loss  | other | (learn rate: 0.008889)
    169   | 28.0%, 0.999571 | 22.8%, 0.682007 * | 27.7%, 0.972044 |
    292   | 28.5%, 1.080433 | 18.1%, 0.521762 * |                 |
    398   | 28.2%, 1.143995 | 15.7%, 0.448233 * |                 |

    """
    def __init__(p):
        p.name = 'e033'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 64
        p.epochs = 10000
        p.final_epochs = 40
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (64, 64)

        p.rng_seed = 13579

        p.learning_rate = 0.08
        p.min_learning_rate = 0.0003
        p.patience = 25
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.90
        p.activation = nn.relu

        p.l1_reg = 0.000
        p.l2_reg = 0.000
        p.dropout_conv = 0.3
        p.dropout_hidd = 0.5

        p.resizer = pp.ContiguousBoxPreserveAspectRatioResizer
        # p.resizer = nn.ContiguousBoxStretchResizer
        p.contiguous_box_threshold = 2

        # p.preprocessor = nn.Rotator360
        p.preprocessor = pp.Canonicalizer

    def build_net(p, n):
        n.add_convolution(32, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(64, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_shape_input()
        n.add_hidden(1024)
        n.add_dropout(p.dropout_hidd)
        n.add_shape_input()
        n.add_hidden(1024)
        n.add_dropout(p.dropout_hidd)
        n.add_shape_input()
        n.add_logistic()


class Experiment032(object):
    """
    comments:  same as e026 except:
    (1) using Canonicalizer
    (2) using ContiguousBoxStretchResizer

    results:
    A little better than e031, but ended up in basically the same place.

    epoch | valid err, loss | train err, loss   | test err, loss  | other | (learn rate: 0.008889)
    168   | 27.8%, 0.991252 | 23.3%, 0.697407   | 28.3%, 0.994610 |
    292   | 28.6%, 1.073077 | 17.6%, 0.501878   |                 |

    Killing it.

    This is sort of interesting.  Neither of my attempts to throw away white space
    helped at all.  I wonder if it would have been different if I had been augmenting
    the network with the cropped shape so it knew how much it was being stretched...
    """
    def __init__(p):
        p.name = 'e032'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 64
        p.epochs = 10000
        p.final_epochs = 40
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (64, 64)

        p.rng_seed = 13579

        p.learning_rate = 0.08
        p.min_learning_rate = 0.0003
        p.patience = 25
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.90
        p.activation = nn.relu

        p.l1_reg = 0.000
        p.l2_reg = 0.000
        p.dropout_conv = 0.3
        p.dropout_hidd = 0.5

        # p.resizer = nn.ThresholdBoxPreserveAspectRatioResizer
        p.resizer = pp.ContiguousBoxStretchResizer
        p.contiguous_box_threshold = 2

        # p.preprocessor = nn.Rotator360
        p.preprocessor = pp.Canonicalizer

    def build_net(p, n):
        n.add_convolution(32, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(64, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_shape_input()
        n.add_hidden(1024)
        n.add_dropout(p.dropout_hidd)
        n.add_shape_input()
        n.add_hidden(1024)
        n.add_dropout(p.dropout_hidd)
        n.add_shape_input()
        n.add_logistic()


class Experiment031(object):
    """
    comments:  same as e026 except:
    (1) using Canonicalizer (huge change)
    (2) using ThresholdBoxStretchResizer

    results:
    Looks like similar results as e030.  Validation score has started
    increasing quite as severely but it looks pretty clear that it is no longer
    tracking training score.
    epoch | valid err, loss | train err, loss   | test err, loss  | other | (learn rate: 0.008889)
    179   | 29.2%, 1.005305 | 24.3%, 0.731116   | 30.9%, 1.074430 |
    324   | 29.1%, 1.087129 | 18.5%, 0.535921 * |                 |

    killing it.
    """
    def __init__(p):
        p.name = 'e031'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 64
        p.epochs = 10000
        p.final_epochs = 40
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (64, 64)

        p.rng_seed = 13579

        p.learning_rate = 0.08
        p.min_learning_rate = 0.0003
        p.patience = 25
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.90
        p.activation = nn.relu

        p.l1_reg = 0.000
        p.l2_reg = 0.000
        p.dropout_conv = 0.3
        p.dropout_hidd = 0.5

        # p.resizer = nn.ThresholdBoxPreserveAspectRatioResizer
        p.resizer = pp.ThresholdBoxStretchResizer
        p.box_threshold = 253

        # p.preprocessor = nn.Rotator360
        p.preprocessor = pp.Canonicalizer

    def build_net(p, n):
        n.add_convolution(32, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(64, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_shape_input()
        n.add_hidden(1024)
        n.add_dropout(p.dropout_hidd)
        n.add_shape_input()
        n.add_hidden(1024)
        n.add_dropout(p.dropout_hidd)
        n.add_shape_input()
        n.add_logistic()

class Experiment030(object):
    """
    comments:  same as e026 except:
    (1) using Canonicalizer (huge change)

    ran it twice.  First time (tagged e030) has no left/right up/down flips.  The second
    time (tagged e030fb) does have those flips.

    results:
    e030: without flips it showed significant improvement at the beginning but then by around
    epoch 30 was starting to overfit a bit, so I killed it and restarted with flips.

    e030f:  for try with flips there is clearly something wrong.  At epoch 128 It's been stuck
    at around 3.4 loss (80% error) almost since the beginning.  Must be a bug.  Fixed
    it with e030fb.

    e030fb: starts over fitting around epoch 187, with best validation loss of 0.942
    and test loss of 0.985289. By epoch 278, back up to 0.999024 with training loss of 0.55.
    killing it.

    epoch | valid err, loss | train err, loss   | test err, loss  | other | (learn rate: 0.008889)
    187   | 27.8%, 0.942312 | 22.1%, 0.662829 * | 27.7%, 0.985289 |
    279   | 27.5%, 0.999024 | 19.1%, 0.550569 * |                 |

    """
    def __init__(p):
        p.name = 'e030'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 64
        p.epochs = 10000
        p.final_epochs = 40
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (64, 64)

        p.rng_seed = 13579

        p.learning_rate = 0.08
        p.min_learning_rate = 0.0003
        p.patience = 25
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.90
        p.activation = nn.relu

        p.l1_reg = 0.000
        p.l2_reg = 0.000
        p.dropout_conv = 0.3
        p.dropout_hidd = 0.5

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer

        # p.preprocessor = nn.Rotator360
        p.preprocessor = pp.Canonicalizer

    def build_net(p, n):
        n.add_convolution(32, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(64, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_shape_input()
        n.add_hidden(1024)
        n.add_dropout(p.dropout_hidd)
        n.add_shape_input()
        n.add_hidden(1024)
        n.add_dropout(p.dropout_hidd)
        n.add_shape_input()
        n.add_logistic()


class Experiment029(object):
    """
    comments:  same as e028 (64x64, big conv/hidden layers,) except:
    (1) more dropout.  (e028 was doing well but started to diverge pretty
    severely, so I killed it and I'm trying again with this.)
    (2) higher minimum learning rate

    results:
    Looks like it was still making significant validation progress
    at .002963 even though it wasn't making training progress any longer and
    dropped learning rates.  But I think it would be worth continuing to train
    it at that learning rate.  It's never seeing the same image twice, so the training
    rate might stall out while the validation rate is still improving.

    Also, this suggests that methods to improve training rate (batch normalization,
    prelu, adadelta) might be worthwhile because maybe a much bigger network
    with higher dropout could push a little further.

    At epoch 567 divergence isn't too severe to I'll let it keep doing it's thing
    and see if it makes any progress.

    TERMINATED BY AMAZON!!!!

    This is the only data I can get after days of training:
    646   | 27.0%, 0.872192 | 28.1%, 0.874499 * |                 |
    647   | 27.4%, 0.869648 | 28.2%, 0.879756 * |                 |
    648   | 27.1%, 0.862260 | 28.1%, 0.864985 * |                 |
    649   | 27.1%, 0.863992 | 28.1%, 0.867609 * |                 |
    650   | 26.7%, 0.853321 | 28.0%, 0.866756 * |                 |
    651   | 27.0%, 0.863217 | 27.9%, 0.863799 * |                 |
    652   | 27.0%, 0.852540 | 27.7%, 0.862395 * |                 |
    653   | 27.2%, 0.854296 | 28.3%, 0.870480 * |                 |
    654   | 26.7%, 0.846113 | 27.9%, 0.869809 * |                 |
    655   | 26.5%, 0.839424 | 28.3%, 0.871263 * |                 |
    656   | 26.5%, 0.838367 | 27.6%, 0.865278 * |                 |
    657   | 26.7%, 0.835137 | 28.0%, 0.865598 * |                 |
    658   | 26.4%, 0.835217 | 28.2%, 0.871606 * |                 |
    659   | 26.6%, 0.834086 | 28.1%, 0.866866 * |                 |
    660   | 26.4%, 0.828096 | 28.0%, 0.863546 * |                 |
    661   | 26.3%, 0.827590 | 27.9%, 0.862918 * |                 |
    662   | 26.2%, 0.831816 | 28.0%, 0.862147 * |                 |
    663   | 26.3%, 0.830877 | 27.8%, 0.858187 * |                 |
    664   | 26.3%, 0.827367 | 28.0%, 0.867163 * |                 |
    665   | 26.0%, 0.820031 | 28.1%, 0.857398 * |                 |
    666   | 25.7%, 0.818368 | 27.8%, 0.861749 * |                 |
    667   | 26.0%, 0.815621 | 27.8%, 0.856138 * |                 |
    668   | 26.6%, 0.822562 | 28.2%, 0.866746 * |                 |
    669   | 26.1%, 0.818723 | 28.0%, 0.858407 * |                 |
    670   | 26.2%, 0.811484 | 27.8%, 0.863605 * |                 |
    671   | 26.2%, 0.812067 | 27.6%, 0.857159 * |                 |
    672   | 25.9%, 0.808138 | 27.8%, 0.852131 * |                 |
    673   | 26.1%, 0.811510 | 27.7%, 0.849469 * |                 |
    674   | 25.8%, 0.807539 | 27.2%, 0.847441 * |                 |
    675   | 26.0%, 0.804366 | 27.9%, 0.854067 * |                 |
    676   | 25.8%, 0.808741 | 27.9%, 0.852438 * |                 |
    677   | 25.4%, 0.792937 | 28.0%, 0.857402 * |                 |
    678   | 26.1%, 0.810321 | 27.9%, 0.849507 * |                 |

    But I'm pretty sure that this was after it started to train the final
    40 epochs so the validation scores are pretty meaningless.  I think
    the best real validation score was 0.89 something (worse than e026's 0.887)
    so I wasn't very hopeful that this was going to produce a better
    leaderboard score.  Still it would have been nice to find out!  Since it
    doesn't seem to have been doing any better, and it took so long to train,
    I'm not going to rerun it... so I'll never know.
    """
    def __init__(p):
        p.name = 'e029'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 64
        p.epochs = 10000
        p.final_epochs = 40
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (64, 64)

        p.rng_seed = 13579

        p.learning_rate = 0.08
        p.min_learning_rate = 0.0003
        p.patience = 25
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.90
        p.activation = nn.relu

        p.l1_reg = 0.000
        p.l2_reg = 0.000
        p.dropout_conv = 0.5
        p.dropout_hidd = 0.6

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360

    def build_net(p, n):
        n.add_convolution(64, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(96, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_shape_input()
        n.add_hidden(1280)
        n.add_dropout(p.dropout_hidd)
        n.add_shape_input()
        n.add_hidden(1280)
        n.add_dropout(p.dropout_hidd)
        n.add_shape_input()
        n.add_logistic()


class Experiment028(object):
    """
    comments:  same as e026 (which is pretty much the same as e020) except:
    (1) more filters in conv layer
    (2) more hidden units

    results:
    looks like it is overfitting nicely, so I'm killing it, and I'll run it again
    with more dropout.
    """
    def __init__(p):
        p.name = 'e028'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 64
        p.epochs = 10000
        p.final_epochs = 40
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (64, 64)

        p.rng_seed = 13579

        p.learning_rate = 0.08
        p.patience = 25
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.90
        p.activation = nn.relu

        p.l1_reg = 0.000
        p.l2_reg = 0.000
        p.dropout_conv = 0.3
        p.dropout_hidd = 0.5

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360

    def build_net(p, n):
        n.add_convolution(64, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(96, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_shape_input()
        n.add_hidden(1280)
        n.add_dropout(p.dropout_hidd)
        n.add_shape_input()
        n.add_hidden(1280)
        n.add_dropout(p.dropout_hidd)
        n.add_shape_input()
        n.add_logistic()


class Experiment027(object):
    """
    comments:  same as e026 except:
    (1) slightly bigger images
    (1) one more conv layer
    (1) more filters in conv layers
    (2) bigger hidden layers


    results:
    Looks like I never really ran this experiment.  Killed it before the first
    epoch because it was taking too long to train.
    """
    def __init__(p):
        p.name = 'e027'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 64
        p.epochs = 10000
        p.final_epochs = 40
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (96, 96)

        p.rng_seed = 13579

        p.learning_rate = 0.08
        p.patience = 15
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.90
        p.activation = nn.relu

        p.l1_reg = 0.000
        p.l2_reg = 0.000
        p.dropout_conv = 0.3
        p.dropout_hidd = 0.5

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360

    def build_net(p, n):
        n.add_convolution(48, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(64, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(96, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_shape_input()
        n.add_hidden(1280)
        n.add_dropout(p.dropout_hidd)
        n.add_shape_input()
        n.add_hidden(1280)
        n.add_dropout(p.dropout_hidd)
        n.add_shape_input()
        n.add_logistic()


class Experiment026(object):
    """
    comments:  same as e020 except:
    (1) p.patience = 15  <NOPE!  FORGOT TO DO THIS!>
    (2) p.final_epochs = 40 (instead of 20)
    (3) p.max_momentum = 0.90 (instead of 0.95)
    (4) I think this might also have a different rule for when to change the
    learning rate

    I want something fairly quick to generate a confusion matrix.
    This is basically a repeat of e020 but with more final training
    and less patience so it will finish faster.

    OOPS.  DIDN'T LOWER PATIENCE

    [running it twice; once before fixing image stretch and once after.
    Compare results]

    results:
    I also ran this on the inverted images and got very similar results by epoch
    133... that seems good though I'm not sure exactly how the math works...

    looks like I'm getting some good old fashioned overfitting here!
    Yay.  I can increase dropout slightly and run it again and probably expect
    to see some improvement.

    The version without the image stretch bug kept training until it was clearly
    over fitting so I stopped it and started it training again on the validation
    set to generate a submission.  The version with the image stretch bug crapped
    out right before training on the validation (due to string formatting error!)
    so I restarted it training on the validation set.  But basically it looks like
    I should add some dropout to this model and performance will improve somewhat.
    The version without the image stretch bug was still improving rapidly enough
    that it was at its original learning rate, so it might have continued for quite
    a while and overfit severly.  It's possible with dropout the validation score
    would get considerably better...
    """
    def __init__(p):
        p.name = 'e026'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 64
        p.epochs = 10000
        p.final_epochs = 40
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (64, 64)

        p.rng_seed = 13579

        p.learning_rate = 0.08
        p.patience = 25
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.90
        p.activation = nn.relu

        p.l1_reg = 0.000
        p.l2_reg = 0.000
        p.dropout_conv = 0.3
        p.dropout_hidd = 0.5

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360

    def build_net(p, n):
        n.add_convolution(32, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(64, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_shape_input()
        n.add_hidden(1024)
        n.add_dropout(p.dropout_hidd)
        n.add_shape_input()
        n.add_hidden(1024)
        n.add_dropout(p.dropout_hidd)
        n.add_shape_input()
        n.add_logistic()


class Experiment025(object):
    """
    comments:  same as e020 except:
    (1) 0.0005 l2_reg instead of 0.0
    (2) p.final_epochs = 40 (instead of 20)
    (3) p.max_momentum = 0.90 (instead of 0.95)

    I noticed that http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf
    used weight decay in addition to dropout, which I think is the same
    thing as l2 regularization.  So I'm curious if it makes any difference
    for my models, and its super easy to try, so I'm trying it.


    results: doesn't look promising so I'm killing it.  I don't see much
    point in l2 as regularization since my training score does not tend to
    depart much from my validation score.  I was mainly running this experiment
    because I was intrigued by the suggestion in the paper above that it helped
    learning.  It doesn't seem to do that (here, at least).

    e025:
    epoch | valid err, loss | train err, loss   | test err, loss  | other | (learn rate: 0.080000)
    026   | 41.7%, 1.443799 | 47.2%, * 1.632537 | 42.5%, 1.493528 |

    e020
    epoch 026 | valid: 39.1%, 1.330937 |
    train: 44.4%, * 1.516527 | learn rate 0.080000, time 202.6 |
    test: 39.7%, 1.387161, time 4.0


    """
    def __init__(p):
        p.name = 'e025'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 64
        p.epochs = 10000
        p.final_epochs = 40
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (64, 64)

        p.rng_seed = 13579

        p.learning_rate = 0.08
        p.patience = 25
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.90
        p.activation = nn.relu

        p.l1_reg = 0.000
        p.l2_reg = 0.0005
        p.dropout_conv = 0.3
        p.dropout_hidd = 0.5

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360

    def build_net(p, n):
        n.add_convolution(32, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(64, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_shape_input()
        n.add_hidden(1024)
        n.add_dropout(p.dropout_hidd)
        n.add_shape_input()
        n.add_hidden(1024)
        n.add_dropout(p.dropout_hidd)
        n.add_shape_input()
        n.add_logistic(121)



class Experiment024(object):
    """
    comments: same as e019 except:
    (0) augment with image shape on all hidden layers
    (1) 50% more conv filters in early layers,
    (2) less max momentum (.9 instead of .95, because that seems to be what
    others have used in some papers I just read, e.g. http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf)
    (3) twice as many final epochs training with validation/test data

    it looks like e019 is going to be my new high score soon.
    So, in the spirit of a genetic algorithm, lets combine the shape augmentation
    of e020 (my current high score, which used image shape on all hidden layers)
    with a net like e019, but with more filters in the convolution layers.

    (I was thinking of adding more convolutional layers but then realized
    that with just one additional layer compared to e020 (i.e. e019), the output
    size with 128x128 is already 4,4 which is the same as e020 with 64x64.
    So there isn't enough room to put another convolution layer with the same
    parameters.  So if I want to experiment with more convolution layers, there are
    perhaps two sensible ways to go about it.
    One is to reduce the max pooling or filter
    shape size.  The other is to double the size of the images again.)

    (If I double the size of the images it seems like that would have a similar effect
    as reducing the filter shape size and max pooling... it seems like the network
    might be able to recognize finer details.  But there might not be much sense in
    doubling the image size because the average image is only 66x77 with a standard
    deviation of 43x49 (so the biggest images that fit within one standard deviation
    are 109x126).  So the vast majority of images are already above or very near full
    resolution at 128x128.  In fact, it might make more sense to decrease the image
    size to speed up training.)

    (Instead of adding more convolutional layers, perhaps the better experiment
    would be to augment the output of the convolutional layers with the
    raw image data so the nn might reconstruct some of the detail lost in the
    conv layers where it needs to.  It seems a bit unlikely that that will help much,
    because there are so many different orientations that images can be in
    and a fully connected layer doesn't have much a of system for making similar
    inferences from totally different parts of the image.  Conv layers don't have
    much of a system for that either.  Perhaps that is a reason to increase the filter
    numbers in the convolutional layers.  So I'm going increasing the number of
    filters in the conv layers slightly.)

    results:
    Training is way too slow, and I'm more excited about other approaches, so I'm
    killing this.  First, I'm not excited about big images, since they are
    so slow to train and because so many of the images are actually smaller.  I think
    64 is actually a pretty nice compromise given my hardware limitations.  Also,
    if I need more detail I'm more interested in throwing away white space right now.
    Second, I'm more excited about my canonicalization idea.  I really think that is going
    to work.


    """
    def __init__(p):
        p.name = 'e024'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 64
        p.epochs = 10000
        p.final_epochs = 40
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (128, 128)

        p.rng_seed = 13579

        p.learning_rate = 0.08
        p.patience = 25
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.90
        p.activation = nn.relu

        p.l1_reg = 0.000
        p.l2_reg = 0.000
        p.dropout_conv = 0.3
        p.dropout_hidd = 0.5

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360

    def build_net(p, n):
        n.add_convolution(48, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(64, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(96, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_shape_input()
        n.add_hidden(2048)
        n.add_dropout(p.dropout_hidd)
        n.add_shape_input()
        n.add_hidden(2048)
        n.add_dropout(p.dropout_hidd)
        n.add_shape_input()
        n.add_logistic()


class Experiment023(object):
    """
    comments: try a little bit more stretch compared to e022.  At epoch 005, e022 is
    getting results similar to e020 (current best score), perhaps very slightly
    better but probably within the realm of chance.  Using e022 as a baseline
    (because it has roughly the same results as e020) if I keep increasing stretch
    gradually (up to to levels in e021) and never see any clear improvement that seems
    like pretty good evidence that it is not helping.

    NOTE THAT THE PREPROCESSOR CODE WAS WRONG SO IMAGES WERE
    MOSTLY RETAINING ASPECT RATIO
    THUS THIS MIGHT PERFORM WORSE THAN e020 FOR THAT REASON

    results:
    so far looks like this amount of stretch is not helping compared to e020.
    maybe I should cancel this experiment and try half the stretch as in e022.
    I'm guessing that will be pretty much the same as e022 and e020 but I would
    pretty much have convinced myself that stretch isn't helping at that point.

    At epoch 037 seems to be worse in every way compared to e020 so I'm killing it.

    e023:
    epoch | valid err, loss | train err, loss   | test err, loss  | other | (learn rate: 0.080000)
    037   | 39.4%, 1.310488 | 42.3%, * 1.405957 | 40.6%, 1.360033 |

    e020:
    epoch 037 | valid: 36.0%, 1.194538 |
    train: 41.4%, * 1.385026 | learn rate 0.080000, time 201.9 |
    test: 36.6%, 1.234509, time 4.0
    """
    def __init__(p):
        p.name = 'e023'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 64
        p.epochs = 10000
        p.final_epochs = 40
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (64, 64)

        p.rng_seed = 13579

        p.learning_rate = 0.08
        p.patience = 25
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.95
        p.activation = nn.relu

        p.l1_reg = 0.000
        p.l2_reg = 0.000
        p.dropout_conv = 0.3
        p.dropout_hidd = 0.5

        # p.resizer = resize_preserve_aspect_ration
        p.resizer = pp.StochasticStretchResizer
        p.stochastic_stretch_range = (.90, 1.10)
        p.preprocessor = pp.Rotator360

    def build_net(p, n):
        n.add_convolution(32, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(64, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_shape_input()
        n.add_hidden(1024)
        n.add_dropout(p.dropout_hidd)
        n.add_shape_input()
        n.add_hidden(1024)
        n.add_dropout(p.dropout_hidd)
        n.add_shape_input()
        n.add_logistic(121)



class Experiment022(object):
    """
    comments: sanity test to make sure I can get good scores when
    stretch range is set to 1,1 (so it should be the same as simple
    StretchResizer).  If not, this suggests there is an implementation
    problem rather than random stretch simply not being a good idea.  If this
    works, then maybe I should try much smaller stretch values.

    Looks like it passed the sanity test at 1,1, so I'm tweaking this experiment
    to .95, 1.05.  The point of stretching (or rotating, or any preprocessing
    transformation) is to make the network generalize better to the validation,
    testing, and ultimately production data.  So what I'm looking for here is
    a reduction in validation error compared to training error.  If validation
    error is high, it seems like stretching is probably not serving a useful purpose.
    It may be that convolution networks don't benefit that much from stretching...
    that is not intuitive to me, but one way I can imagine that in this case
    is if they are following edges around a shape, a slightly bigger object will
    not change it's edge very much so stretching slightly won't have much effect,
    but stretching more moves the shape into an unrealistic range?  Or perhaps
    if the network is recognizing textures rather than curves, stretching...  I guess
    the obvious answer is that if there is enough real variation in size in the
    untransformed data, then all stretching can do is introduce unrealistic variation.

    results:
    NOTE THAT THE PREPROCESSOR CODE WAS WRONG SO IMAGES WERE
    MOSTLY RETAINING ASPECT RATIO
    THUS THIS MIGHT PERFORM WORSE THAN e020 FOR THAT REASON

    At epoch 003 it looks like there might be a slight improvement over experiment
    e020.

    At epoch 49 there does not appear to be any improvement over e020 so I'm killing it.
    I'm concluding that stretching the images is pretty pointless.  The most important
    thing to do next is look at a confusion matrix and see if I can get any insight
    into the kinds of errors my networks are making and how to fix them.

    e022:
    epoch | valid err, loss | train err, loss   | test err, loss  | other | (learn rate: 0.080000)
    048   | 36.1%, 1.170343 | 39.8%, * 1.310867 | 37.1%, 1.236377 |
    049   | 36.6%, 1.189121 | 39.4%, * 1.299105 |                 |

    e020:
    epoch 049 | valid: 33.7%, 1.125033 |
    train: 39.2%, * 1.286366 | learn rate 0.080000, time 201.6 |
    test: 35.1%, 1.176390, time 4.0
    """
    def __init__(p):
        p.name = 'e022'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 64
        p.epochs = 10000
        p.final_epochs = 40
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (64, 64)

        p.rng_seed = 13579

        p.learning_rate = 0.08
        p.patience = 25
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.95
        p.activation = nn.relu

        p.l1_reg = 0.000
        p.l2_reg = 0.000
        p.dropout_conv = 0.3
        p.dropout_hidd = 0.5

        # p.resizer = resize_preserve_aspect_ration
        p.resizer = pp.StochasticStretchResizer
        p.stochastic_stretch_range = (.95, 1.05)
        p.preprocessor = pp.Rotator360

    def build_net(p, n):
        n.add_convolution(32, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(64, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_shape_input()
        n.add_hidden(1024)
        n.add_dropout(p.dropout_hidd)
        n.add_shape_input()
        n.add_hidden(1024)
        n.add_dropout(p.dropout_hidd)
        n.add_shape_input()
        n.add_logistic(121)


class Experiment021(object):
    """
    comments: same as e020 (my current high score), but with StochasticStretchResizer,
    my first experiment using stochastic stretch.

    Stretch range of .75 to 1.25 seems to make things worse.  Next try much
    smaller values.

    results:
    NOTE THAT THE PREPROCESSOR CODE WAS WRONG SO IMAGES WERE
    MOSTLY RETAINING ASPECT RATIO
    THUS THIS MIGHT PERFORM WORSE THAN e020 FOR THAT REASON

    Seems to be mostly stalled out.

    From the start validation score was worse relative to e020, and it kept
    getting worse from there.  Learning the training set was also slower, so
    doesn't seem like there is much to be gained by keeping this experiment running.

    killing it.

    epoch | valid err, loss | train err, loss   | test err, loss  | other | (learn rate: 0.080000)
    -----------------------------------------------------------------------------------------------------
    000   | 79.7%, 3.460341 | 83.7%, * 3.618303 | 80.0%, 3.538742 | train time: 207.6, test time: 4.2,
    001   | 73.4%, 2.938830 | 73.1%, * 2.886436 | 74.0%, 3.004934 |
    002   | 71.1%, 2.752540 | 68.5%, * 2.591671 | 72.7%, 2.794086 |
    epoch | valid err, loss | train err, loss   | test err, loss  | other | (learn rate: 0.008889)
    186   | 39.9%, 1.353098 | 32.2%,   1.004887 | 39.8%, 1.385378 |
    190   | 42.1%, 1.430638 | 32.0%,   0.997546 |                 |

    """
    def __init__(p):
        p.name = 'e021'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 64
        p.epochs = 10000
        p.final_epochs = 40
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (64, 64)

        p.rng_seed = 13579

        p.learning_rate = 0.08
        p.patience = 25
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.95
        p.activation = nn.relu

        p.l1_reg = 0.000
        p.l2_reg = 0.000
        p.dropout_conv = 0.3
        p.dropout_hidd = 0.5

        # p.resizer = resize_preserve_aspect_ration
        p.resizer = pp.StochasticStretchResizer
        p.stochastic_stretch_range = (0.75, 1.25)
        p.preprocessor = pp.Rotator360

    def build_net(p, n):
        n.add_convolution(32, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(64, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_shape_input()
        n.add_hidden(1024)
        n.add_dropout(p.dropout_hidd)
        n.add_shape_input()
        n.add_hidden(1024)
        n.add_dropout(p.dropout_hidd)
        n.add_shape_input()
        n.add_logistic(121)


class Experiment020_repeat(object):
    """
    comments:  trying to reproduce e020 results with revised nn

    trying it two ways with shapes normalized by the mean (as they were in e020)
    and with the std (as they should be)

    results: So far results of both are comparable to e020 (finally!!).
    Looks like the problem was that I was squaring the images before rotating while
    preparing them for training and that led the pictures to be differently sized
    compared to the validation and test sets.

    since they seem to be tracking e020 pretty closely, I'm cancelling this experiment.
    But I think I've learning something important.  Always repeat an experiment whenever
    I make big changes as a quick check to make sure I haven't broken anything.

    If I hadn't repeated this experiment I might not have caught the problem that was
    causing worse results, EVER.

    """
    def __init__(p):
        p.name = 'e020-repeat'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 64
        p.epochs = 10000
        p.final_epochs = 20
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (64, 64)

        p.rng_seed = 13579

        p.learning_rate = 0.08
        p.patience = 25
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.95
        p.activation = nn.relu

        p.l1_reg = 0.000
        p.l2_reg = 0.000
        p.dropout_conv = 0.3
        p.dropout_hidd = 0.5

        # p.resizer = resize_preserve_aspect_ration
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360

    def build_net(p, n):
        n.add_convolution(32, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(64, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_shape_input()
        n.add_hidden(1024)
        n.add_dropout(p.dropout_hidd)
        n.add_shape_input()
        n.add_hidden(1024)
        n.add_dropout(p.dropout_hidd)
        n.add_shape_input()
        n.add_logistic(121)


class Experiment020(object):
    """
    comments:  Augment with shape info on each layer.  Otherwise, the same as
    experiment 11, my highest scoring experiment.
    This is what e018 was meant to be but I accidentally used 128x128 images in e018.
    I'm reluctant to cancel e018 though because it looks like it is doing well.

    SHAPES WERE IMPROPERLY NORMALIZED: (shapes - mean) / mean instead of (shapes - mean) / std

    results:
    """
    def __init__(p):
        p.name = 'e020'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 64
        p.epochs = 10000
        p.final_epochs = 20
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (64, 64)

        p.rng_seed = 13579

        p.learning_rate = 0.08
        p.patience = 25
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.95
        p.activation = nn.relu

        p.l1_reg = 0.000
        p.l2_reg = 0.000
        p.dropout_conv = 0.3
        p.dropout_hidd = 0.5

        # p.resizer = resize_preserve_aspect_ration
        p.resizer = pp.StretchResizer
        p.preprocessor = pp.Rotator360

    def build_net(p, n):
        n.add_convolution(32, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(64, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_shape_input()
        n.add_hidden(1024)
        n.add_dropout(p.dropout_hidd)
        n.add_shape_input()
        n.add_hidden(1024)
        n.add_dropout(p.dropout_hidd)
        n.add_shape_input()
        n.add_logistic(121)


class Experiment019(object):
    """
    comments: the same as e017 (4 times bigger *images* than e015) but
    (1)with an additional convolution layer and (2) lower dropout on
    convolutional layers

    IMAGES THIS BIG CAUSE PROBLEMS WHEN MAKING PREDICTIONS BECAUSE THE ENTIRE
    SUBMISSION SET CANNOT BE TRANSFORMED ALL AT ONCE.  HAD TO WRITE
    SOME CUSTOM CODE TO GET THIS TO WORK.

    SHAPES WERE IMPROPERLY NORMALIZED: (shapes - mean) / mean instead of (shapes - mean) / std

    results:
    This seems to be doing substantially better than e017, and has quicker epochs too.
    Maybe what I should be doing is adding more convolutional layers... until the
    size of the network feeding into the hidden layers is about the same as with a
    64x64 image...

    Anyway, it ran into serious memory issues when generating the submission set
    because of the image size.  It took be forever to finally generate a set.  When
    I finally did this became my new high score by a hair and moved me up to 91 on the
    leader board.  But given how slow the models are to train I think the more fruitful
    avenue will be smaller image sizes.

    """
    def __init__(p):
        p.name = 'e019'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 64
        p.epochs = 10000
        p.final_epochs = 20
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (128, 128)

        p.rng_seed = 13579

        p.learning_rate = 0.08
        p.patience = 25
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.95
        p.activation = nn.relu

        p.l1_reg = 0.000
        p.l2_reg = 0.000
        p.dropout_conv = 0.3
        p.dropout_hidd = 0.5

        # p.resizer = resize_preserve_aspect_ration
        p.resizer = plankton.resize_stretch
        p.preprocessor = nn.Rotator360

    def build_net(p, n):
        n.add_convolution(32, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(64, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(64, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_shape_input()
        n.add_hidden(2048)
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(2048)
        n.add_dropout(p.dropout_hidd)
        n.add_logistic(121)


class Experiment018(object):
    """
    comments:  Augment with shape info on each layer.
    Otherwise, the same as e011,
    my highest scoring experiment, except
    (1)hidden layers are twice as big.
    (2)images are 4 times bigger (2 times in each dimension)
    (3) each hidden layer is augmented with input shape

    Also, very similar to e017, except
    (1) conv dropout is .3 instead of .5
    (2) each hidden layer is augmented with input shape instead of just the first

    SHAPES WERE IMPROPERLY NORMALIZED: (shapes - mean) / mean instead of (shapes - mean) / std

    results:
    Looks like it was doing marginally better than e017, but since e017
    was itself not that impressive I don't see much point in continuing training
    this since its so slow.

    epoch 103 | valid: 27.9%, 0.943377 |
    train: 24.3%, * 0.730852 | learn rate 0.026667, time 1214.5 |
    test: 28.7%, 0.936091, time 21.1
    epoch 104 | valid: 27.9%, 0.937261 |
    train: 24.6%, ! 0.732342 | learn rate 0.026667, time 1213.5 |
    test: 29.2%, 0.941346, time 21.1
    epoch 147 | valid: 28.6%, 0.974284 |
    train: 23.6%, ! 0.676700 | learn rate 0.026667, time 1213.7
    set momentum to 0.838844358921

    """
    def __init__(p):
        p.name = 'e018'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 64
        p.epochs = 10000
        p.final_epochs = 20
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (128, 128)

        p.rng_seed = 13579

        p.learning_rate = 0.08
        p.patience = 25
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.95
        p.activation = nn.relu

        p.l1_reg = 0.000
        p.l2_reg = 0.000
        p.dropout_conv = 0.3
        p.dropout_hidd = 0.5

        # p.resizer = resize_preserve_aspect_ration
        p.resizer = plankton.resize_stretch
        p.preprocessor = nn.Rotator360

    def build_net(p, n):
        n.add_convolution(32, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(64, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_shape_input()
        n.add_hidden(2048)
        n.add_dropout(p.dropout_hidd)
        n.add_shape_input()
        n.add_hidden(2048)
        n.add_dropout(p.dropout_hidd)
        n.add_shape_input()
        n.add_logistic(121)


class Experiment017(object):
    """
    comments: 4 times bigger *images* than e015
    each epoch takes 1217 seconds.

    SHAPES WERE IMPROPERLY NORMALIZED: (shapes - mean) / mean instead of (shapes - mean) / std

    results:
    At epoch 54, it's looking substantially improved over e015, with a 1.079 validation loss compared to 1.199.
    I wonder what adding another convolutional layer at this size would do.

    So far it looks like only a very minor improvement over e015, if any.

    So after days of training it doesn't seem to be any better than e015.  By
    epoch 217 the best test score is a little better but the best validation score
    is not.

    I'm shutting this down.  Much bigger images do not seem to make a huge difference,
    at least with this architecture.

    e017:
    epoch 193 | valid: 29.3%, 0.9822467 |
    train: 24.7%, ! 0.7130108 | learn rate 0.00290, time 1219.2 |
    test: 30.1%, 0.9717435, time 21.5
    epoch 217 | valid: 29.4%, 1.008561 |
    train: 23.3%, ! 0.6721258 | learn rate 0.00290, time 1217.9

    e015:
    epoch 215 | valid: 29.9%, 0.9782749 |
    train: 30.4%, ! 0.9300431 | learn rate 0.00290, time 205.2 |
    test: 30.1%, 1.015653, time 4.3
    """
    def __init__(p):
        p.name = 'e017'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 64
        p.epochs = 10000
        p.final_epochs = 20
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (128, 128)

        p.rng_seed = 13579

        p.learning_rate = 0.08
        p.patience = 25
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.95
        p.activation = nn.relu

        p.l1_reg = 0.000
        p.l2_reg = 0.000
        p.dropout_conv = 0.5
        p.dropout_hidd = 0.5

        # p.resizer = resize_preserve_aspect_ration
        p.resizer = plankton.resize_stretch
        p.preprocessor = nn.Rotator360

    def build_net(p, n):
        n.add_convolution(32, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(64, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_shape_input()
        n.add_hidden(2048)
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(2048)
        n.add_dropout(p.dropout_hidd)
        n.add_logistic(121)


class Experiment016(object):
    """
    comments: 4 times bigger *hidden layers* than e015.
    each epoch takes 245 seconds.

    Looks like this hasn't improved things over e015.  In fact it's made things slightly worse.
    It's still not done yet though.  Yup.  No improvement here.  At least with the
    other parameter settings, hidden layer size does not seem to be a performance bottleneck.

    SHAPES WERE IMPROPERLY NORMALIZED: (shapes - mean) / mean instead of (shapes - mean) / std

    results:
    epoch 283 | valid: 29.0%, 0.9847562 | train: 26.9%, ! 0.8012653 | learn rate 0.00030, time 246.3 |
    test: 29.0%, 1.018726, time 7.9
    epoch 355 | valid: 29.0%, 0.9909822 | train: 26.2%, * 0.7766649 | learn rate 3.65790, time 246.1
    best validation error 28.9832746479% (0.984756231308); trained for 1819.25642398 minutes

    """
    def __init__(p):
        p.name = 'e016'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 64
        p.epochs = 10000
        p.final_epochs = 20
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (64, 64)

        p.rng_seed = 13579

        p.learning_rate = 0.08
        p.patience = 25
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.95
        p.activation = nn.relu

        p.l1_reg = 0.000
        p.l2_reg = 0.000
        p.dropout_conv = 0.5
        p.dropout_hidd = 0.5

        # p.resizer = resize_preserve_aspect_ration
        p.resizer = plankton.resize_stretch
        p.preprocessor = nn.Rotator360

    def build_net(p, n):
        n.add_convolution(32, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(64, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_shape_input()
        n.add_hidden(8192)
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(8192)
        n.add_dropout(p.dropout_hidd)
        n.add_logistic(121)


class Experiment015(object):
    """
    comments: my first experiment augmenting the hidden layers with the dimensions
    of the pre-resized images!  It seems like this should make a big difference.
    Let's see how it does...  Same params as e014 (except for augmentation with image shape)

    SHAPES WERE IMPROPERLY NORMALIZED: (shapes - mean) / mean instead of (shapes - mean) / std

    results:
    epoch 331 | valid: 29.3%, 0.9617513 | train: 28.3%, ! 0.8482558 | learn rate 0.00010, time 205.1 |
    test: 30.4%, 1.003180, time 4.3
    epoch 378 | valid: 29.4%, 0.9662026 | train: 27.6%, * 0.8290898 | learn rate 1.21930, time 205.1
    best validation error 29.3133802817% (0.961751282215); trained for 1553.89745175 minutes
    """
    def __init__(p):
        p.name = 'e015'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 64
        p.epochs = 10000
        p.final_epochs = 20
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (64, 64)

        p.rng_seed = 13579

        p.learning_rate = 0.08
        p.patience = 25
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.95
        p.activation = nn.relu

        p.l1_reg = 0.000
        p.l2_reg = 0.000
        p.dropout_conv = 0.5
        p.dropout_hidd = 0.5

        # p.resizer = resize_preserve_aspect_ration
        p.resizer = plankton.resize_stretch
        p.preprocessor = nn.Rotator360

    def build_net(p, n):
        n.add_convolution(32, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(64, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_shape_input()
        n.add_hidden(2048)
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(2048)
        n.add_dropout(p.dropout_hidd)
        n.add_logistic(121)


class Experiment014(object):
    """
    comments: even bigger hidden layers (2048) when compared to e013 (1536)

    results:
    epoch 290 | valid: 30.4%, 0.9947055 | train: 30.5%, ! 0.9285978 | learn rate 0.00030, time 201.8 |
    test: 30.6%, 1.008973, time 4e+00
    epoch 330 | valid: 30.5%, 0.9973828 | train: 30.0%, * 0.9206767 | learn rate 3.65790, time 201.5
    best validation error 30.4357394366% (0.994705498219); trained for 1464.47301905 minutes
    """
    def __init__(p):
        p.name = 'e014'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 64
        p.epochs = 10000
        p.final_epochs = 20
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (64, 64)

        p.rng_seed = 13579

        p.learning_rate = 0.08
        p.patience = 25
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.95
        p.activation = nn.relu

        p.l1_reg = 0.000
        p.l2_reg = 0.000
        p.dropout_conv = 0.5
        p.dropout_hidd = 0.5

        # p.resizer = resize_preserve_aspect_ration
        p.resizer = plankton.resize_stretch
        p.preprocessor = nn.Rotator360

    def build_net(p, n):
        n.add_convolution(32, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(64, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_hidden(2048)
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(2048)
        n.add_dropout(p.dropout_hidd)
        n.add_logistic(121)


class Experiment013(object):
    """
    comments: trying slightly bigger hidden layers and slightly higher dropout.
    Lowering the dropout in e012 does not seem to have helped the validation score so I'm
    setting it to the what seems like a pretty typical 0.5.  However my networks
    aren't learning the training set.  Perhaps there is just too much variation
    in the training set with all the rotations, so it will never be able to learn it.
    But perhaps the networks is just not big enough.  Lets see.

    results:
    epoch 276 | valid: 30.7%, 1.004352 | train: 31.5%, ! 0.9773416 | learn rate 0.00030, time 198.7 |
    test: 30.6%, 1.010020, time 4e+00
    epoch 364 | valid: 31.1%, 1.007447 | train: 31.5%, * 0.9656191 | learn rate 1.21930, time 199.6
    best validation error 30.6998239437% (1.00435209274); trained for 1449.28048735 minutes
    """
    def __init__(p):
        p.name = 'e013'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 64
        p.epochs = 10000
        p.final_epochs = 20
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (64, 64)

        p.rng_seed = 13579

        p.learning_rate = 0.08
        p.patience = 25
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.95
        p.activation = nn.relu

        p.l1_reg = 0.000
        p.l2_reg = 0.000
        p.dropout_conv = 0.5
        p.dropout_hidd = 0.5

        # p.resizer = resize_preserve_aspect_ration
        p.resizer = plankton.resize_stretch
        p.preprocessor = nn.Rotator360

    def build_net(p, n):
        n.add_convolution(32, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(64, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_hidden(1536)
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(1536)
        n.add_dropout(p.dropout_hidd)
        n.add_logistic(121)


class Experiment012(object):
    """
    comments: trying lower dropout compared to experiments 9 and 11.  If this
    doesn't work I'll try higher dropout.  It seems like adding all those rotations
    should decrease the need for dropout.  Lets see.  These aren't terribly
    good score.  Looks like dropout is doing some good work here.  Training
    error has diverge from validation error and still looks like its on a downward
    trajectory.  So it's probably just overfitting now.  That reinforces my suspicion
    that I'll get better results from e013 with higher dropout and larger hidden
    layers. When dropout is high it's doing a good job of keeping the train and validation loss
    together but the neural network is not powerful enough to get the train loss down.

    results: seems to have stalled out:
    epoch 60, validation error 31.1179577465% (1.05402600765), train error 33.9185855263 % (1.06165277958), learning rate 0.0799999982119, time 194.300886154
    test error 32.1331521739% (1.07964348793), time 3.84876585007
    epoch 85, validation error 30.5897887324% (1.00896036625), train error 30.1069078947 % (0.91680753231), learning rate 0.0266666654497, time 194.505093813
    test error 29.5516304348% (1.02085101604), time 3.83501219749
    epoch 88, validation error 29.4674295775% (0.985477268696), train error 28.2935855263 % (0.860040664673), learning rate 0.0266666654497, time 194.295869827
    test error 29.9592391304% (1.0024433136), time 3.84153294563
    epoch 189, validation error 29.0052816901% (1.02256083488), train error 21.8708881579 % (0.628086984158), learning rate 0.00296296272427, time 194.183541059
    epoch 190, validation error 28.0589788732% (1.03070414066), train error 22.4629934211 % (0.639359176159), learning rate 0.00296296272427, time 195.70521903
    epoch 231, validation error 28.0149647887% (1.03677749634), train error 19.2927631579 % (0.538920223713), learning rate 0.000987654202618, time 194.648644924
    epoch 232, validation error 28.5651408451% (1.04403626919), train error 19.2722039474 % (0.53795593977), learning rate 0.000987654202618, time 194.789721966
    """
    def __init__(p):
        p.name = 'e012'
        p.num_images = None
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.num_submission_images = None

        p.batch_size = 64
        p.epochs = 10000
        p.final_epochs = 20
        p.image_shape = (64, 64)

        p.rng_seed = 13579

        p.learning_rate = 0.08
        p.patience = 25
        p.improvement_threshold = 0.995
        p.momentum = 0.5
        p.max_momentum = 0.95
        p.activation = nn.relu

        p.l1_reg = 0.000
        p.l2_reg = 0.000
        p.dropout_conv = 0.2
        p.dropout_hidd = 0.3

        p.resizer = plankton.resize_stretch
        p.preprocessor = nn.Rotator360

    def build_net(p, n):
        n.add_convolution(32, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(64, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_convolution(128, (5, 5), (2, 2))
        n.add_dropout(p.dropout_conv)
        n.add_hidden(1024)
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(1024)
        n.add_dropout(p.dropout_hidd)
        n.add_logistic(121)


if __name__ == '__main__':
    # plankton.make_confusion_matrix_from_saved_network(Experiment037())
    experiment = Experiment()
    tools.save_output(plankton.OUTPUT_DIR + os.sep + experiment.name + '-output.txt',
                      plankton.run_experiment, experiment)