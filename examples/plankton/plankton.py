__author__ = 'isaac'

from pydnn import nn
from pydnn import preprocess as pp
from pydnn import tools
from pydnn import data
from pydnn import img_util

import numpy as np
import pandas as pd
from scipy.misc import imread

import os
from os.path import join
import time

config = tools.load_config('PLANKTON_CONFIG', __file__, 'plankton.conf')['plankton']
train_set = data.DirectoryLabeledImageSet(config['input_train'], config['dtype'])
test_set = data.UnlabeledImageSet(config['input_test'])


def write_submission_csv_file(file_name, probs, image_file_names):
    import gzip
    df = pd.DataFrame(data=probs, index=image_file_names, columns=train_set.get_labels())
    df.index.name = 'image'
    with gzip.open(file_name, 'w') as outFile:
        df.to_csv(outFile)


def generate_submission_file(net, name, num=None):
        if num is None:
            num = 130400

        if num < 130400:
            batch_size = num
            num_batches = 1
        else:
            batch_size = 16300
            num_batches = 8

        probabilities = []
        files = []
        dotter = tools.Dot()
        print('generating probabilities...')
        for i in range(num_batches):
            fns, images, = test_set.build(i * batch_size,
                                                (i + 1) * batch_size)
            _, probs = net.predict({'images': images})
            probabilities.append(probs)
            files += fns
            dotter.dot(str(i) + ' ')
        dotter.stop()
        probabilities = np.row_stack(probabilities)
        print('writing csv file...')
        write_submission_csv_file(name, probabilities, files)


def load_net_and_generate_submission_file(net_name, submission_name):
    print('loading net')
    net = nn.load(net_name)
    generate_submission_file(net, submission_name)
# n = 'e0??'
# load_net_and_generate_submission_file(n + '_best_net.pkl', n + '_sub_best.csv.gz')
# load_net_and_generate_submission_file(n + '_final_net.pkl', n + '_sub_final.csv.gz')


def write_confusion_matrices_to_csv_files(experiment, num_images, matrices):
    set_names = ['train', 'valid', 'test']
    labels = train_set.get_labels()
    files, given_labels = zip(*train_set.get_files(num_images))
    for (matrix, mistakes), set_name in zip(matrices, set_names):
        df = pd.DataFrame(matrix, index=labels, columns=labels)
        df.to_csv(join(config['output'], experiment + '_conf_mtrx_' + set_name + '.csv'))

        file_indices, right_indices, wrong_indices = zip(*mistakes)
        file_names = [files[index] for index in file_indices]
        right_labels = [given_labels[index] for index in file_indices]
        wrong_labels = [labels[index] for index in wrong_indices]
        df = pd.DataFrame({'wrong': wrong_labels, 'right': right_labels},
                          index=file_names)
        df.to_csv(join(config['output'], experiment + '_mistakes_' + set_name + '.csv'))


def make_confusion_matrix_from_saved_network(e):
    print('making confusion matrices...')
    data = train_set.build(e.num_images)
    data = pp.split_training_data(data, e.batch_size, e.train_pct, e.valid_pct, e.test_pct)
    net = nn.load(join(config['input_post'], e.name + '_best_net.pkl'))
    net.preprocessor.set_data(data)
    write_confusion_matrices_to_csv_files(e.name, net.get_confusion_matrices())
    print('...done making confusion matrices')


def analyze_confusion_matrix(matrix_file):
    n = 121
    rng = np.random.RandomState(123)
    x = pd.read_csv(matrix_file, index_col=0)
    data = np.index_exp[:n, :n]


    x['total'] = x.iloc[data].sum(axis=1)
    total_predictions = x['total'].sum()
    values = x.iloc[data].values    # values can sometimes return a copy
    np.fill_diagonal(values, 0)  # so must save, zero and reassign
    x.iloc[data] = values           # (I've discovered after some confusion)
    x['bad'] = x.iloc[data].sum(axis=1)
    total_bad = x['bad'].sum()
    x['pct_bad'] = x['bad'] / x['total']

    top_by_num = x.sort('total', ascending=False)[0:10].index.values
    worst_by_num = x.sort('bad', ascending=False)[0:10].index.values
    worst_by_num_ct = x.sort('bad', ascending=False)[0:10].values
    worst_by_pct = x.sort('pct_bad', ascending=False)[0:10].index.values
    worst_by_pct_ct = x.sort('pct_bad', ascending=False)[0:10].values

    print("total predictions: {}".format(total_predictions))
    print("total bad predictions: {}".format(total_bad))

    print("most common classes (regardless of error rate): " + str(top_by_num))

    def most_confused_with(c):
        # get the row, and only the class values (not the generated columns)
        row = x.loc[c]
        row = row.iloc[:n]
        row.sort(ascending=False)

        last_non_zero = 10
        # print(row.iloc[:10].values)
        for index, z in enumerate(row.iloc[:last_non_zero].values):
            if z <= 0:
                last_non_zero = index
                break

        # return the top classes for the row
        return zip(row.iloc[:last_non_zero].index.values, row.iloc[:last_non_zero].values)

    def print_worst(classes):
        for c in classes:
            c_total = x.loc[c, 'total']
            c_bad = x.loc[c, 'bad']
            c_contribution_to_error = float(c_bad) / total_bad
            c_fair_share_of_error = c_total / total_predictions
            print('\nclass {}:'.format(c))
            print('total predictions: {}'.format(c_total))
            print('total bad predictions: {}'.format(c_bad))
            print('fair share of error: {:.3%}'.format(c_fair_share_of_error))
            print('contribution to error: {:.3%} ({:.3f} time fair share)'.format(
                c_contribution_to_error, c_contribution_to_error / c_fair_share_of_error))
            print('most often confused with' + str(most_confused_with(c)))

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.image as mpimg

    def show_worst(worst):
        def add_row(directory, count, index):
            first = True
            for i in range(5):
                sub = fig.add_subplot(11, 5, index)
                fn = rng.choice(tools.get_files(join(config['input_train'], directory),
                                                  cache=True))
                image = mpimg.imread(fn)
                plt.imshow(image, interpolation='none', cmap=cm.Greys_r)
                if first:
                    title = '{}: {} ({}x{})'.format(
                        directory, count, image.shape[0], image.shape[1])
                    first = False
                else:
                    title = '({}x{})'.format(image.shape[0], image.shape[1])
                sub.set_title(title, size=10)

                sub.axis('off')
                index += 1
            return index

        for c in worst:
            fig = plt.figure()
            pos = add_row(c, x.loc[c, 'bad'], 1)
            for i, num in most_confused_with(c):
                pos = add_row(i, num, pos)
            plt.show()

    print("---------- worst classes by number -----------")
    print_worst(worst_by_num)
    show_worst(worst_by_num)
    print("---------- worst classes by percent ----------")
    print_worst(worst_by_pct)
    show_worst(worst_by_num)

    print('might also be useful to look at the post transformed images to gain'
          'insight into why the net is not able to recognize them well')

    print('also remember to look at whether confusion is symmetrical (i.e. if A '
          'is frequently confused for B, is B also frequently confused for A?)')

    print('at some point might be worth looking at the specific image that were'
          'incorrectly classified, but to begin with Im just looking for the most'
          'important trends (classes with the most confusion) and individual images'
          'shouldnt tell me too much')


def run_experiment(e):
    print('############## {} ################'.format(e.name))
    print('start time: ' + tools.now())
    rng = np.random.RandomState(e.rng_seed)

    data = train_set.build(e.num_images)
    data = pp.split_training_data(data, e.batch_size, e.train_pct, e.valid_pct, e.test_pct)

    if e.resizer == pp.StochasticStretchResizer:
        resizer = e.resizer(rng, e.stochastic_stretch_range)
    elif e.resizer in [pp.ThresholdBoxPreserveAspectRatioResizer,
                       pp.ThresholdBoxStretchResizer]:
        resizer = e.resizer(e.box_threshold)
    elif e.resizer in [pp.ContiguousBoxPreserveAspectRatioResizer,
                       pp.ContiguousBoxStretchResizer]:
        resizer = e.resizer(e.contiguous_box_threshold)
    else:
        resizer = e.resizer()

    preprocessor = e.preprocessor(data, e.image_shape, resizer, rng, config['dtype'])
    net = nn.NN(preprocessor=preprocessor,
                channel='images',
                num_classes=121,
                batch_size=e.batch_size,
                rng=rng,
                activation=e.activation,
                name=e.name,
                output_dir=config['output'])
    e.build_net(net)

    try:
        net.train(
            updater=e.learning_rule,
            epochs=e.epochs,
            final_epochs=e.final_epochs,
            l1_reg=e.l1_reg,
            l2_reg=e.l2_reg)
    finally:
        print('Experiment "{}" ended'.format(e.name))

    print('generating probabilities based on final network...')
    generate_submission_file(net,
                             join(config['output'], e.name + '_submission_final.csv.gz'),
                             e.num_submission_images)

    net = nn.load(join(config['output'], e.name + '_best_net.pkl'))

    print('generating probabilities based on best network...')
    generate_submission_file(net,
                             join(config['output'], e.name + '_submission_best.csv.gz'),
                             e.num_submission_images)

    print('generating and writing confusion matrix based on best network...')
    net.preprocessor.set_data(data)
    write_confusion_matrices_to_csv_files(e.name, e.num_images,
                                          net.get_confusion_matrices())

    print('end time: ' + tools.now())
    return net


def average_submissions(in_files, weights=None):
    import gzip
    subs = []
    for f in in_files:
        print('loading ' + f)
        with gzip.open(join(config['input_post'], f), 'r') as inFile:
            subs.append(np.loadtxt(
                fname=inFile,
                dtype=config['dtype'],
                delimiter=',',
                skiprows=1,
                usecols=range(1, 122)))
    # avg = np.mean(subs, axis=0)
    avg = np.average(subs, axis=0, weights=weights)
    out_file = (join(config['input_post'], 'avg_probs_' +
                time.strftime("%Y-%m-%d--%H-%M-%S") + '.csv.gz'))
    print('saving...')
    write_submission_csv_file(out_file, avg, test_set.get_files())
    print('done')


def show_mistakes(mistakes_file):
    mistakes = pd.read_csv(mistakes_file, index_col=0)
    for index, row in mistakes.iterrows():
        images = [imread(join(config['input_train'], row['right'],
                         os.path.basename(index)))]

        right_images = np.random.choice(
            tools.get_files(join(config['input_train'], row['right'])),
            9, replace=False)
        wrong_images = np.random.choice(
            tools.get_files(join(config['input_train'], row['wrong'])),
            10, replace=False)

        images.extend([imread(fn) for fn in right_images])
        images.extend([imread(fn) for fn in wrong_images])
        print(os.path.basename(index), row['right'], row['wrong'])
        img_util.show_images_as_tiles(images, size=(128, 128), canvas_dims=(4, 5))
#show_mistakes(SUBMISSION_DIR + '/results/e075_mistakes_valid.csv')