# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This file is a modified version from the code source
# https://github.com/facebookresearch/dcd

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from .distributions import Categorical  
from .common import *
from collections import OrderedDict

class CantileverNetwork(DeviceAwareModule):
    """
    Actor-Critic module 
    """
    def __init__(self, 
        observation_space, 
        action_space,
        recurrent_arch='lstm',
        recurrent_hidden_size=256, 
        actor_fc_layers=(32, 32),
        value_fc_layers=(32, 32),
        random_z_dim=0,
        random=False):        
        super().__init__()

        self.random = random
        self.action_space = action_space
        num_actions = action_space.n
        
        # MultiBinary embeddings
        if type(observation_space) == OrderedDict:
            # If the observation space is a dictionary, we need to handle it differently
            obs_shape = observation_space['node_binary'].shape
            obs_len = obs_shape[0]
            self.preprocessed_input_size = obs_len
            self.preprocessed_input_size += random_z_dim
            self.base_output_size = self.preprocessed_input_size
        else:
            # If the observation space is a single tensor, we can use its shape directly
            obs_shape = observation_space.shape[0]
            self.preprocessed_input_size = obs_shape
            self.base_output_size = self.preprocessed_input_size

        # RNN
        self.rnn = None
        if recurrent_arch:
            self.rnn = RNN(
                input_size=self.preprocessed_input_size, 
                hidden_size=recurrent_hidden_size,
                arch=recurrent_arch)
            self.base_output_size = recurrent_hidden_size

        # Policy head
        self.actor = nn.Sequential(
            make_fc_layers_with_hidden_sizes(actor_fc_layers, input_size=self.base_output_size),
            Categorical(actor_fc_layers[-1], num_actions)
        )

        # Value head
        self.critic = nn.Sequential(
            make_fc_layers_with_hidden_sizes(value_fc_layers, input_size=self.base_output_size),
            init_(nn.Linear(value_fc_layers[-1], 1))
        )

        apply_init_(self.modules())

        self.train()

    @property
    def is_recurrent(self):
        return self.rnn is not None

    @property
    def recurrent_hidden_state_size(self):
        # """Size of rnn_hx."""
        if self.rnn is not None:
            return self.rnn.recurrent_hidden_state_size
        else:
            return 0

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def _forward_base(self, inputs, rnn_hxs, masks):
        # Extract tensor data from the inputs dictionary
        if isinstance(inputs, dict):  # Ensure inputs is a dictionary
            in_z = inputs.get('random_z', torch.tensor([], device=self.device))
            in_node_binary = inputs.get('node_binary', torch.tensor([], device=self.device))
            in_embedded = torch.cat((in_node_binary, in_z), dim=-1)
        else:
            in_embedded = inputs

        if self.rnn is not None:
            core_features, rnn_hxs = self.rnn(in_embedded, rnn_hxs, masks)
        else:
            core_features = in_embedded

        return core_features, rnn_hxs

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        if self.random:
            if isinstance(inputs, dict):  # Ensure inputs is a dictionary
                B = inputs['node_binary'].shape[0]
            else:
                B = inputs.shape[0]
            action = torch.zeros((B,1), dtype=torch.int64, device=self.device)
            values = torch.zeros((B,1), device=self.device)
            action_log_dist = torch.ones(B, self.action_space.n, device=self.device)
            for b in range(B):
                action[b] = self.action_space.sample()

            return values, action, action_log_dist, rnn_hxs

        core_features, rnn_hxs = self._forward_base(inputs, rnn_hxs, masks)
        dist = self.actor(core_features)
        value = self.critic(core_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_dist = dist.logits

        return value, action, action_log_dist, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        core_features, rnn_hxs = self._forward_base(inputs, rnn_hxs, masks)
        return self.critic(core_features)

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, return_policy_logits=False):
        core_features, rnn_hxs = self._forward_base(inputs, rnn_hxs, masks)

        dist = self.actor(core_features)
        value = self.critic(core_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        if return_policy_logits:
            return value, action_log_probs, dist_entropy, rnn_hxs, dist
        
        return value, action_log_probs, dist_entropy, rnn_hxs