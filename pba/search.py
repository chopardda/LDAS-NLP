"""Run PBA Search."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
import ray
from ray.tune import run_experiments
from ray.tune.schedulers import PopulationBasedTraining
import tensorflow as tf

from pba.setup import create_hparams
from pba.setup import create_parser
from pba.train import RayModel


def main(_):
    FLAGS = create_parser("search")
    hparams = create_hparams("search", FLAGS)
    hparams_config = hparams.values()

    train_spec = {
        "run": RayModel,
        "resources_per_trial": {
            "cpu": FLAGS.cpu,
            "gpu": FLAGS.gpu
        },
        "stop": {
            "training_iteration": hparams.num_epochs,
        },
        "config": hparams_config,
        "local_dir": FLAGS.local_dir,
        "checkpoint_freq": FLAGS.checkpoint_freq,
        "num_samples": FLAGS.num_samples
    }

    if FLAGS.restore:
        train_spec["restore"] = FLAGS.restore

    def explore(config):
        """Custom explore function.

    Args:
      config: dictionary containing ray config params.

    Returns:
      Copy of config with modified augmentation policy.
    """
        new_params = []
        if config["explore"] == "default_explore_fct":
            for i, param in enumerate(config["hp_policy"]):
                if random.random() < 0.2: # --- resample rapam uniformly from domain
                    if i % 2 == 0:
                        new_params.append(random.uniform(0,1)) # probability
                    else:
                        new_params.append(random.uniform(0,1)) # magnitude
                else:
                    amt = np.random.uniform(0,0.1)
                    if random.random() < 0.5:
                        new_params.append(max(0, param - amt)) # --- perturb original value, clamp values to min 0
                    else:
                        new_params.append(min(1, param + amt)) # --- perturb original value, clamp values to max 1
        else:
            raise ValueError()
        config["hp_policy"] = new_params
        return config

    ray.init()

    # TODO: set for specific data set
    if hparams.dataset == 'cola':
        reward_attribute = 'val_matthews'
    else:
        reward_attribute = 'val_acc'

    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        reward_attr=reward_attribute,
        perturbation_interval=FLAGS.perturbation_interval,
        custom_explore_fn=explore,
        log_config=True)

    run_experiments(
        {
            FLAGS.name: train_spec
        },
        scheduler=pbt,
        reuse_actors=False,
        verbose=True,
    )


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
