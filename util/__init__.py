# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This file is a modified version from the code source
# https://github.com/facebookresearch/dcd

import glob
import os
import shutil
import collections
import timeit
import random

import numpy as np
import torch
from torchvision import utils as vutils

from envs.registration import make as gym_make
from envs.cantilever.pyansys_sim import Cantilever
from .make_agent import make_agent
from .filewriter import FileWriter
from .eff_indep import effective_independence
from envs.wrappers import ParallelAdversarialVecEnv, VecMonitor, VecNormalize, \
    VecPreprocessImageWrapper, FlattenMultiDiscreteActions


class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)
        self.__dict__ = self

#Convert array to csv
def array_to_csv(a):
    return ','.join([str(v) for v in a])

#Print method
def cprint(condition, *args, **kwargs):
    if condition:
        print(*args, **kwargs)


def init(module, weight_init, bias_init, gain=1):
    """ Initialise the weight and bias of the module
    using 
    weight_init function and gain of default 1
    bias_init function
    """
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def safe_checkpoint(state_dict, path, index=None, archive_interval=None):
    filename, ext = os.path.splitext(path) # ex. ('test/env_step/eve', '.py')
    path_tmp = f'{filename}_tmp{ext}'
    torch.save(state_dict, path_tmp)

    os.replace(path_tmp, path)

    #This is to archeive files that were temporarily store.
    if index is not None and archive_interval is not None and archive_interval > 0:
        if index % archive_interval == 0:
            archive_path = f'{filename}_{index}{ext}'
            shutil.copy(path, archive_path)


def cleanup_log_dir(log_dir, pattern='*'):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, pattern))
        for f in files:
            os.remove(f)

def seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def save_images(images, path=None, normalize=False, channels_first=False):
    if path is None:
        return

    if isinstance(images, (list, tuple)):
        images = torch.tensor(np.stack(images), dtype=torch.float)
    elif isinstance(images, np.ndarray):
        images = torch.tensor(images, dtype=torch.float)

    if normalize:
        images = images/255

    if not channels_first:
        if len(images.shape) == 4:
            images = images.permute(0,3,1,2)
        else:
            images = images.permute(2,0,1)

    grid = vutils.make_grid(images)
    vutils.save_image(grid, path)


def get_obs_at_index(obs, i):
    if isinstance(obs, dict):
        return {k: obs[k][i] for k in obs.keys()}
    else:
        return obs[i]


def set_obs_at_index(obs, obs_, i):
    if isinstance(obs, dict):
        for k in obs.keys():
            obs[k][i] = obs_[k].squeeze(0)
    else:
        obs[i] = obs_[0].squeeze(0)

#Check if action space is discret
def is_discrete_actions(env, adversary=False):
    if adversary:
        return env.adversary_action_space.__class__.__name__ == 'Discrete'
    else:
        return env.action_space.__class__.__name__ == 'Discrete'


def _make_env(args):
    env_kwargs = {}
    if args.singleton_env:
        env_kwargs.update({
            'fixed_environment': True})
    env_kwargs.update({
        'sim_modes': args.sim_modes,
        'num_sensors': args.num_sensors,
        'pyansys_env': args.pyansys_env,
    })
    env = gym_make(args.env_name, **env_kwargs)
    env = FlattenMultiDiscreteActions(env)
    return env



def create_parallel_env(args, adversary=True):
    make_fn = lambda: _make_env(args)
    venv = ParallelAdversarialVecEnv([make_fn]*args.num_processes, adversary=adversary)
    venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
    venv = VecNormalize(venv=venv, ob=False, ret=args.normalize_returns)
    venv = VecPreprocessImageWrapper(venv=venv)
    ued_venv = venv

    if args.singleton_env:
        seeds = [args.seed]*args.num_processes
    else:
        seeds = [i for i in range(args.num_processes)]
    venv.set_seed(seeds)

    return venv, ued_venv


def is_dense_reward_env(env_name):
    if env_name.startswith('Cantilever'):
        return True
    else:
        return False


def make_plr_args(args, obs_space, action_space):
    return dict( 
        seeds=[], 
        obs_space=obs_space, 
        action_space=action_space, 
        num_actors=args.num_processes,
        strategy=args.level_replay_strategy,
        replay_schedule=args.level_replay_schedule,
        score_transform=args.level_replay_score_transform,
        temperature=args.level_replay_temperature,
        eps=args.level_replay_eps,
        rho=args.level_replay_rho,
        replay_prob=args.level_replay_prob, 
        alpha=args.level_replay_alpha,
        staleness_coef=args.staleness_coef,
        staleness_transform=args.staleness_transform,
        staleness_temperature=args.staleness_temperature,
        sample_full_distribution=args.train_full_distribution,
        seed_buffer_size=args.level_replay_seed_buffer_size,
        seed_buffer_priority=args.level_replay_seed_buffer_priority,
        use_dense_rewards=is_dense_reward_env(args.env_name),
        gamma=args.gamma
    )
class envs_params(Cantilever):
    def __init__(self, config):
        # Correctly call the parent class's __init__ method
        super().__init__(config)
    def get_data(self):
        data = {}
        data['observation_space_node'] = self.observation_space_node
        data["coord_2d_array"] = self.coord_2d_array
        
        data['left_edge_nodes'] = self.left_edge_nodes
        data['right_edge_nodes'] = self.right_edge_nodes
        data['top_edge_nodes'] = self.top_edge_nodes
    
        data['phi_data'] = self.phi_data
        data['correlation_covariance_matrix_data'] = self.correlation_covariance_matrix_data
        data['encode_data'] = self.encode_data
        
        # Store the data as an instance attribute
        return data
