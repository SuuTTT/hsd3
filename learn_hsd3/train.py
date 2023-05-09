# Homework.py

# Hey there, future master of Reinforcement Learning! Ready to code your way to glory once again?
# This time, we're pulling out the big guns. More imports, more variables, and a lot more fun!

# First, as usual, let's import some modules. Don't worry, they won't bite.
import itertools
import json
import logging
import os
import shutil
from copy import copy
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List

import gym
import hydra
import numpy as np
import torch as th
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from visdom import Visdom

# Remember hucc, our best friend from the last homework? Yeah, it's back!
import hucc
from hucc.agents.utils import discounted_bwd_cumsum_
from hucc.hashcount import HashingCountReward
from hucc.spaces import th_flatten

log = logging.getLogger(__name__)

# Here's our class again, this time with some new friends. 
# I'll leave you to figure out their purpose. Remember, curiosity is the key to success!
### Class: TrainingSetup

#Your class should have the following attributes:

# * `cfg`: A DictConfig object. This is your configuration dictionary.
# * `agent`: An instance of the hucc.Agent class.
# * `model`: A nn.Module object. This is your model.
# * `tbw`: A SummaryWriter instance.
# * `viz`: A Visdom instance. This will be used for visualization.
# * `rq`: An instance of hucc.RenderQueue.
# * `envs`: An instance of hucc.VecPyTorch. This is your environment.
# * `eval_envs`: Another instance of hucc.VecPyTorch. This is your evaluation environment.
# * `eval_fn`: A Callable object. It's a function that takes in a TrainingSetup and an integer and returns nothing.
# * `n_samples`: An integer. This keeps count of the samples.
# * `replaybuffer_checkpoint_path`: A string, default is 'replaybuffer.pt'. This is the path for the replay buffer checkpoint.
# * `training_state_path`: A string, default is 'training\_state.json'. This is the path for the training state.
# * `hcr_checkpoint_path`: A string, default is 'hashcounts.pt'. This is the path for the hash counts reward checkpoint.
# * `hcr_space`: For this assignment, leave this as None.
# * `hcr`: An instance of HashingCountReward. For this assignment, leave this as None.

# ### Method: close

# Your `close` method should do the following:

# * Close `rq`, `envs`, and `eval_envs`.
# * Try to unlink (delete) the file at the path `replaybuffer_checkpoint_path`. If the file doesn't exist, don't do anything.



# Class we're going to complete:
class TrainingSetup(SimpleNamespace):
    # cfg: This variable will hold a configuration dictionary. 
    # It's like a treasure map, but instead of leading us to a chest full of gold, it guides us through our training process.
    cfg: DictConfig

    # agent: This is our hero, the one who's going to learn and explore the environment.
    # It's like a small child, eager to learn and discover the world.
    agent: hucc.Agent

    # model: This is the brain of our agent. It's going to make our agent smarter... or dumber, depending on how well we train it.
    model: nn.Module

    # tbw: This will be used for TensorBoard to visualize our training progress.
    # It's like our agent's diary, where it writes down everything it learns.
    tbw: SummaryWriter

    # And here come the rest of our adventure tools! I'll leave you to find out what they do.
    viz: hucc.Visdom
    rq: hucc.RenderQueue
    envs: hucc.VecPyTorch
    eval_envs: hucc.VecPyTorch
    eval_fn: Callable  # Callable[[TrainingSetup, int], None]
    n_samples: int = 0
    replaybuffer_checkpoint_path: str = 'replaybuffer.pt'
    training_state_path: str = 'training_state.json'
    hcr_checkpoint_path: str = 'hashcounts.pt'
    hcr_space = None
    hcr: hucc.HashingCountReward = None

    # Now, here's the first challenge for you!
    # Complete this close function to close all the necessary resources.
    # Remember, just like you need to turn off the lights before leaving a room,
    # you need to close everything before ending your training.
    def close(self):
        # TODO: Use the hucc documentation and your brilliant mind to figure out how to close the resources.
        # Don't forget to handle exceptions! They're like the monsters in your adventure, and you're the hero who's going to defeat them.
        pass

# Welcome back, brave adventurer! Now, it's time to setup training. 
# Like a boss preparing for an epic quest, we need to ensure our gear (code) is ready for the journey ahead.


# Let's create a function to set up the training. 
# This function will take a configuration dictionary as an argument and return a TrainingSetup object.
def setup_training(cfg: DictConfig) -> TrainingSetup:
    # TODO: Check if CUDA is available. If it's not, then let's use the CPU.
    # Don't forget to set the seed for PyTorch to ensure the results are reproducible.
    # Tip: You might want to use 'th.manual_seed()' for setting the seed.

    # TODO: Setup the Visdom visualization tool.
    # It's like a magical mirror that lets us see how our agent is performing.
    # Remember to use the configuration parameters from 'cfg'!

    # TODO: Create the environment for training and evaluation using hucc.
    # It's like the world where our agent will live and learn. Make sure it's a nice place!

    # TODO: Setup the observation and action spaces.
    # This defines what our agent can see and do in its world.

    # TODO: Create the model. You'll need to recursively create models for each key if the observation and action spaces are dicts.

    # TODO: Use the 'hucc.make_agent()' function to create the agent.
    # It's like bringing our hero to life!

    # TODO: Setup TensorBoard SummaryWriter.
    # It's like a diary where our agent will write down its thoughts and feelings (well, more like performance metrics, but you get the idea).

    # TODO: Setup the HashingCountReward (hcr), if required by the config.
    # It's like a special reward that our agent can get for exploring new things.

    # TODO: Finally, return a TrainingSetup object with all the components we've created.
    # It's like the backpack filled with all the gear our agent will need for its epic quest!

    return TrainingSetup()

# Now, over to you! Pack your bags, and let's set off on this coding adventure!

# Homework.py

# Welcome back, brave adventurer! Now, it's time to evaluate our agent. 
# It's like sending our hero into the coliseum to show off its skills!

# Let's import some modules first. You're a pro at this by now!
from typing import List, Dict, Any
from torch import Tensor
from collections import defaultdict
import numpy as np
import torch as th
import itertools

# Let's create a function to evaluate our agent's performance. 
# This function will take a TrainingSetup object and the number of samples to test on.

def eval(setup: TrainingSetup, n_samples: int = -1):
    # TODO: Extract the agent, request queue, and evaluation environments from the setup.

    # TODO: Reset the environments and initialize variables for collecting rewards, 
    # dones, request queue inputs, images, and metrics.

    # TODO: Now, let's step into the coliseum! Run a loop until all environments have completed.

        # TODO: If video collection is enabled, collect and annotate the video frames.

        # TODO: Get the agent's action and apply it to the environments.

        # TODO: If entropy is being tracked, append it to the entropy_ds list.

        # TODO: If certain metrics are being tracked, append them to the metrics_v dictionary.

        # TODO: Append the reward and done tensors to their respective lists.

        # TODO: If all environments are done, break the loop. Otherwise, reset the done environments.

    # TODO: Compute the undiscounted and discounted returns, and the episode length.

    # TODO: Update the metrics_v dictionary with the computed returns and episode lengths.

    # TODO: Log the metrics to TensorBoard using the agent's summary writer.

    # TODO: Log the average episode length and return.

    # TODO: If entropy was being tracked, log its mean and histogram to TensorBoard.

    # TODO: If video was being collected, annotate the frames with the accumulated reward 
    # and push them to the request queue for display.

# Now, over to you! Strap on your armor, and let's step into the coliseum!
