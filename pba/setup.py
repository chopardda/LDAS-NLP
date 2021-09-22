"""Parse flags and set up hyperparameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import random
import tensorflow as tf

from pba.augmentation_transforms_hp import NUM_HP_TRANSFORM


def create_parser(state):
    """Create arg parser for flags."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name',
        default='bert',
        choices=('simple_rnn', 'bert'))
    parser.add_argument(
        '--data_path',
        default='/tmp/datasets/',
        help='Directory where dataset is located.')
    parser.add_argument(
        '--dataset',
        default='imdb_reviews',
        choices=('cola', 'sst2', 'mrpc', 'stsb', 'qqp', 'mnli_m', 'mnli_mm', 'qnli', 'rte', 'wnli'))
    parser.add_argument(
        '--recompute_dset_stats',
        action='store_true',
        help='Instead of using hardcoded mean/std, recompute from dataset.')
    parser.add_argument('--local_dir', type=str, default='/tmp/ray_results/',  help='Ray directory.')
    parser.add_argument('--restore', type=str, default=None, help='If specified, tries to restore from given path.')
    parser.add_argument('--train_size', type=int, default=7500, help='Number of training examples.')
    parser.add_argument('--val_size', type=int, default=17500, help='Number of validation examples.')
    parser.add_argument('--checkpoint_freq', type=int, default=50, help='Checkpoint frequency.')
    parser.add_argument(
        '--cpu', type=float, default=4, help='Allocated by Ray')
    parser.add_argument(
        '--gpu', type=float, default=1, help='Allocated by Ray')
    parser.add_argument(
        '--aug_policy',
        type=str,
        default='default_aug_policy',
        help=
        'which augmentation policy to use (in augmentation_transforms_hp.py)')
    # search-use only
    parser.add_argument(
        '--explore',
        type=str,
        default='default_explore_fct',
        help='which explore function to use')
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='Number of epochs, or <=0 for default')
    parser.add_argument(
        '--development',
        action='store_true',
        help='evaluate on dev set'
    )
    parser.add_argument(
        '--expsize',
        type=str,
        default='FULL',
        help='Experiment size',
        choices=('1500', '2000', '3000', '4000', 'FULL')
    )
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--wd', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--bs', type=int, default=32, help='batch size')
    parser.add_argument('--max_seq_length', type=int, default=128, help='max seq length')
    parser.add_argument('--test_bs', type=int, default=64, help='test batch size')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of Ray samples')

    if state == 'train':
        parser.add_argument(
            '--use_hp_policy',
            action='store_true',
            help='otherwise use autoaug policy')
        parser.add_argument(
            '--hp_policy',
            type=str,
            default=None,
            help='either a comma separated list of values or a file')
        parser.add_argument(
            '--hp_policy_epochs',
            type=int,
            default=200,
            help='number of epochs/iterations policy trained for')
        parser.add_argument(
            '--no_aug',
            action='store_true',
            help='no additional augmentation at all (besides cutout if not toggled)'
        )
        parser.add_argument(
            '--flatten',
            action='store_true',
            help='randomly select aug policy from schedule')
        parser.add_argument('--name', type=str, default='autoaug')

    elif state == 'search':
        parser.add_argument('--perturbation_interval', type=int, default=10)
        parser.add_argument('--name', type=str, default='augmentation_schedule_search')
    else:
        raise ValueError('unknown state')
    args = parser.parse_args()
    tf.logging.info(str(args))
    return args


def create_hparams(state, FLAGS):  # pylint: disable=invalid-name
    """Creates hyperparameters to pass into Ray config.

  Different options depending on search or eval mode.

  Args:
    state: a string, 'train' or 'search'.
    FLAGS: parsed command line flags.

  Returns:
    tf.hparams object.
  """
    epochs = 0
    tf.logging.info('data path: {}'.format(FLAGS.data_path))
    hparams = tf.contrib.training.HParams(
        train_size=FLAGS.train_size,
        validation_size=FLAGS.val_size,
        dataset=FLAGS.dataset,
        data_path=FLAGS.data_path,
        expsize=FLAGS.expsize,
        batch_size=FLAGS.bs,
        max_seq_length = FLAGS.max_seq_length,
        gradient_clipping_by_global_norm=0.1,
        explore=FLAGS.explore,
        aug_policy=FLAGS.aug_policy,
        recompute_dset_stats=FLAGS.recompute_dset_stats,
        lr=FLAGS.lr,
        weight_decay_rate=FLAGS.wd,
        test_batch_size=FLAGS.test_bs)

    if state == 'train':
        hparams.add_hparam('no_aug', FLAGS.no_aug)
        hparams.add_hparam('development', FLAGS.development)
        hparams.add_hparam('use_hp_policy', FLAGS.use_hp_policy)
        hparams.add_hparam('limit_test_data', False)
        if FLAGS.use_hp_policy:
            if FLAGS.hp_policy == 'random':
                tf.logging.info('RANDOM SEARCH')
                parsed_policy = []
                for i in range(NUM_HP_TRANSFORM * 2):
                    if i % 2 == 0:
                        parsed_policy.append(random.random()) # --- probability
                    else:
                        parsed_policy.append(random.random()) # --- magnitude
            elif FLAGS.hp_policy.endswith('.txt') or FLAGS.hp_policy.endswith(
                    '.p'):
                # --- will be loaded in in data_utils
                parsed_policy = FLAGS.hp_policy
            else:
                # --- parse input into a fixed augmentation policy
                print(FLAGS.hp_policy)
                print(type(FLAGS.hp_policy))
                parsed_policy = FLAGS.hp_policy.split(',')
                parsed_policy = [float(p) for p in parsed_policy]
            hparams.add_hparam('hp_policy', parsed_policy)
            hparams.add_hparam('hp_policy_epochs', FLAGS.hp_policy_epochs)
            hparams.add_hparam('flatten', FLAGS.flatten)
    elif state == 'search':
        hparams.add_hparam('no_aug', False)
        hparams.add_hparam('development', FLAGS.development)
        hparams.add_hparam('use_hp_policy', True)
        hparams.add_hparam('limit_test_data', True)
        hparams.add_hparam('hp_policy',
                           [0 for _ in range(2 * NUM_HP_TRANSFORM)])  # --- default start values of 0
    else:
        raise ValueError('unknown state')

    # -- Add new model here
    if FLAGS.model_name == 'bert':
        hparams.add_hparam('model_name', 'bert')
    else:
        raise ValueError('Not Valid Model Name: %s' % FLAGS.model_name)
    if FLAGS.epochs > 0:
        tf.logging.info('overwriting with custom epochs')
        epochs = FLAGS.epochs
    hparams.add_hparam('num_epochs', epochs)
    tf.logging.info('epochs: {}, lr: {}, wd: {}'.format(
        hparams.num_epochs, hparams.lr, hparams.weight_decay_rate))
    return hparams
