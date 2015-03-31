import unittest
from examples.plankton import plankton

from pydnn import nn
from pydnn import preprocess as pp
from os.path import join


class QuickTest(object):
    def __init__(p):
        p.name = 'quick_test'
        p.batch_size = 4
        # careful changing image shape... can cause theano exceptions
        # with convolution networks
        p.image_shape = (64, 64)
        p.num_images = p.batch_size * 40
        p.num_submission_images = 10
        # p.num_submission_images = None
        p.epochs = 30
        p.final_epochs = 2
        p.train_pct = 80
        p.valid_pct = 15
        p.test_pct = 5
        p.rng_seed = 13579

        p.annealer = nn.WackyLearningRateAnnealer(
            learning_rate=0.08,
            min_learning_rate=0.008,
            patience=1,
            train_improvement_threshold=0.9,
            valid_improvement_threshold=0.95,
            reset_on_decay=None)
        p.learning_rule = nn.Momentum(
            initial_momentum=0.5,
            max_momentum=0.5,
            learning_rate=p.annealer)

        p.activation = nn.relu
        p.l1_reg = 0.000
        p.l2_reg = 0.000
        p.dropout_conv = 0.3
        p.dropout_hidd = 0.5

        p.batch_normalize = False

        # p.resizer = nn.PreserveAspectRatioResizer
        p.resizer = pp.StochasticStretchResizer
        # p.resizer = pp.StretchResizer
        p.stochastic_stretch_range = (0.5, 1.5)
        p.preprocessor = pp.Rotator360

    def build_net(p, n):
        n.add_convolution(8, (5, 5), (10, 10), (4, 4), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        n.merge_data_channel('shapes')
        n.add_hidden(256, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_logistic()


class QuickTest1(QuickTest):
    def __init__(p):
        QuickTest.__init__(p)
        p.resizer = pp.PreserveAspectRatioResizer
        p.name = 'quick_test_1'
        p.epochs = 1
        p.final_epochs = 0
        p.learning_rule = nn.StochasticGradientDescent(p.annealer)

    def build_net(p, n):
        """ test net without augmentation or drop out """
        n.add_convolution(4, (5, 5), (10, 10))
        n.add_mlp(32)


class QuickTest2(QuickTest1):
    def __init__(p):
        QuickTest1.__init__(p)
        p.batch_size = 4
        p.num_images = p.batch_size * 20
        # p.num_images = None
        p.preprocessor = pp.Canonicalizer
        p.resizer = pp.ThresholdBoxStretchResizer
        p.name = 'quick_test_2'
        p.epochs = 10
        p.final_epochs = 2
        p.rng_seed = 128928
        p.box_threshold = 253


class QuickTest3(QuickTest2):
    def __init__(p):
        QuickTest2.__init__(p)
        p.resizer = pp.ContiguousBoxStretchResizer
        p.name = 'quick_test_3'
        p.epochs = 10
        p.final_epochs = 2
        p.contiguous_box_threshold = 2


class QuickTest4(QuickTest3):
    def __init__(p):
        QuickTest3.__init__(p)
        p.name = 'quick_test_4'

    def build_net(p, n):
        """ test net without augmentation or drop out """
        n.add_dropout(0.5)
        n.add_convolution(4, (5, 5), (10, 10))
        n.add_mlp(32)


class QuickTest5(QuickTest):
    def __init__(p):
        QuickTest.__init__(p)
        p.name = 'quick_test_5'
        p.resizer = pp.StretchResizer
        p.num_images = None
        p.epochs = 10
        p.final_epochs = 1
        p.train_pct = 100
        p.valid_pct = 0
        p.test_pct = 0

        p.activation = nn.tanh

    def build_net(p, n):
        """ test net without augmentation or drop out """
        n.add_mlp(4)


class QuickTest6(QuickTest):
    def __init__(p):
        QuickTest.__init__(p)
        p.name = 'quick_test_6'
        p.resizer = pp.StretchResizer
        p.epochs = 10
        p.final_epochs = 1
        p.annealer = nn.LearningRateDecay(
            learning_rate=0.08,
            decay=.03)
        p.learning_rule = nn.AdaDelta(
            rho=0.95,
            epsilon=1e-6,
            learning_rate=p.annealer)

        p.activation = nn.sigmoid

    def build_net(p, n):
        """ test net without augmentation or drop out """
        n.add_mlp(4)


class QuickTest7(QuickTest6):
    def __init__(p):
        super(QuickTest7, p).__init__()
        p.name = 'quick_test_7'
        p.epochs = 1
        p.final_epochs = 1
        p.image_shape = (2, 2)
        p.annealer = nn.LearningRateDecay(
            learning_rate=0.08,
            decay=.03,
            min_learning_rate=0.000000008)
        p.learning_rule = nn.StochasticGradientDescent(p.annealer)

    def build_net(p, n):
        n.add_mlp(1)


class QuickTest8(QuickTest7):
    def __init__(p):
        super(QuickTest8, p).__init__()
        p.name = 'quick_test_8'
        p.epochs = 100
        p.final_epochs = 2
        p.image_shape = (2, 2)
        p.batch_size = 2
        p.num_images = p.batch_size * 20

        p.annealer = nn.WackyLearningRateAnnealer(
            learning_rate=0.08,
            min_learning_rate=0.000000008,
            patience=2,
            train_improvement_threshold=0.9,
            valid_improvement_threshold=0.95,
            reset_on_decay=None)
        p.learning_rule = nn.Momentum(
            initial_momentum=0.5,
            max_momentum=0.6,
            learning_rate=p.annealer)

        p.activation = nn.prelu


class QuickTest9(QuickTest):
    def __init__(p):
        super(QuickTest9, p).__init__()
        p.name = 'quick_test_9'
        p.image_shape = (32, 32)
        p.num_images = p.batch_size * 20
        p.num_submission_images = 10
        p.epochs = 2
        p.final_epochs = 0

        p.batch_normalize = True

        p.resizer = pp.StochasticStretchResizer
        # p.resizer = pp.StretchResizer
        p.stochastic_stretch_range = (0.5, 1.5)
        p.preprocessor = pp.Rotator360PlusGeometry

    def build_net(p, n):
        n.add_conv_pool(8, (5, 5), (10, 10), (4, 4), use_bias=False)
        n.merge_data_channel('geometry')
        n.add_batch_normalization()
        n.add_nonlinearity(nn.relu)
        n.add_dropout(p.dropout_conv)
        n.add_fully_connected(8, weight_init=nn.relu, use_bias=False)
        n.merge_data_channel('geometry')
        n.add_batch_normalization()
        n.add_nonlinearity(nn.relu)
        n.add_dropout(p.dropout_hidd)
        n.add_logistic()


class QuickTest10(QuickTest):
    def __init__(p):
        super(QuickTest10, p).__init__()
        p.name = 'quick_test_10'
        p.learning_rule = nn.Adam(
            b1=0.9,
            b2=0.999,
            e=1e-8,
            lmbda=1 - 1e-8,
            learning_rate=0.001)


class QuickTest11(QuickTest8):
    def __init__(p):
        super(QuickTest11, p).__init__()
        p.annealer = nn.WackyLearningRateAnnealer(
            learning_rate=0.08,
            min_learning_rate=0.000000008,
            patience=2,
            train_improvement_threshold=0.9,
            valid_improvement_threshold=0.95,
            reset_on_decay='validation')
        p.learning_rule = nn.Momentum(
            initial_momentum=0.5,
            max_momentum=0.6,
            learning_rate=p.annealer)


class QuickTest12(QuickTest7):
    def __init__(p):
        super(QuickTest7, p).__init__()
        p.name = 'quick_test_12'
        p.epochs = 8
        p.final_epochs = 2

        p.learning_rule = nn.StochasticGradientDescent(
            learning_rate=nn.LearningRateSchedule(
                schedule=((0, .4),
                          (2, .2),
                          (3, .1),
                          (7, .05))))


class QuickTest13(QuickTest):
    def __init__(p):
        super(QuickTest13, p).__init__()
        p.name = 'quick_test_13'

    def build_net(p, n):
        n.add_convolution(8, (5, 5), (5, 5), (4, 4), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        pathways = n.split_pathways(3)
        for pw in pathways + [n]:
            pw.add_convolution(8, (5, 5), (10, 10), (4, 4), batch_normalize=p.batch_normalize)
            pw.merge_data_channel('shapes')
            pw.add_hidden(8, batch_normalize=p.batch_normalize)
        n.merge_pathways(pathways)
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(8, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_logistic()


class QuickTest14(QuickTest):
    def __init__(p):
        super(QuickTest14, p).__init__()
        p.name = 'quick_test_14'

    def build_net(p, n):
        n.add_convolution(8, (5, 5), (5, 5), (4, 4), batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_conv)
        pathway = n.split_pathways()
        for pw in [pathway, n]:
            pw.add_convolution(8, (5, 5), (10, 10), (4, 4), batch_normalize=p.batch_normalize)
            pw.merge_data_channel('shapes')
            pw.add_hidden(8, batch_normalize=p.batch_normalize)
        n.merge_pathways(pathway)
        n.add_dropout(p.dropout_hidd)
        n.add_hidden(8, batch_normalize=p.batch_normalize)
        n.add_dropout(p.dropout_hidd)
        n.add_logistic()


class TestGeneral(unittest.TestCase):
    def test_prediction_with_saved_reloaded_net(self):
        plankton.run_experiment(QuickTest())
        print('testing prediction with saved/reloaded net...')
        net = nn.load(join(plankton.config['output'], 'quick_test_best_net.pkl'))
        print('loading submission data')
        files, images = plankton.test_set.build(0, 10)
        print('generating probabilities')
        preds, probs_all = net.predict({'images': images})
        print(preds, probs_all)

    def test_continue_training_with_save_reloaded_net(self):
        plankton.run_experiment(QuickTest1())
        print('testing continue training with saved/reloaded net...')

        e = QuickTest1()
        data = plankton.train_set.build(32 * 40)
        data = pp.split_training_data(data, 32, 80, 10, 10)

        net = nn.load(join(plankton.config['output'], 'quick_test_1_best_net.pkl'))

        net.preprocessor.set_data(data)

        net.train(
            updater=e.learning_rule,
            epochs=e.epochs,
            final_epochs=e.final_epochs,
            l1_reg=e.l1_reg,
            l2_reg=e.l2_reg)

    def test_various_experiment_setting_combinations(self):
        import theano
        print('theano version' + theano.__version__)

        plankton.run_experiment(QuickTest2())
        plankton.run_experiment(QuickTest3())
        plankton.run_experiment(QuickTest4())
        plankton.run_experiment(QuickTest5())
        plankton.run_experiment(QuickTest6())
        plankton.run_experiment(QuickTest7())
        plankton.run_experiment(QuickTest8())
        plankton.run_experiment(QuickTest9())
        plankton.run_experiment(QuickTest10())
        plankton.run_experiment(QuickTest11())
        plankton.run_experiment(QuickTest12())
        plankton.run_experiment(QuickTest13())
        plankton.run_experiment(QuickTest14())