# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This file is a modified version from the code source
# https://github.com/facebookresearch/dcd

import argparse

import torch

from util import str2bool


parser = argparse.ArgumentParser(description='RL')


# PPO & other optimization arguments. 
parser.add_argument(
    '--algo',
    type=str,
    default='ppo',
    choices=['ppo'],
    help='Which RL algorithm to use.')
parser.add_argument(
    '--lr', 
    type=float, 
    default=1e-4, 
    help='Learning rate')
parser.add_argument(
    '--eps',
    type=float,
    default=1e-5,
    help='RMSprop optimizer epsilon.')
parser.add_argument(
    '--alpha',
    type=float,
    default=0.99,
    help='RMSprop optimizer alpha.')
parser.add_argument(
    '--gamma',
    type=float,
    default=0.995,
    help='Discount factor for rewards.')
parser.add_argument(
    '--use_gae',
    type=str2bool, nargs='?', const=True, default=True,
    help='Use generalized advantage estimator.')
parser.add_argument(
    '--gae_lambda',
    type=float,
    default=0.95,
    help='GAE lambda parameter.')
parser.add_argument(
    '--entropy_coef',
    type=float,
    default=0.0,
    help='Entropy bonus coefficient for student.')
parser.add_argument(
    '--adv_entropy_coef',
    type=float,
    default=0.0,
    help='Entropy bonus coefficient for teacher.')
parser.add_argument(
    '--value_loss_coef',
    type=float,
    default=0.5,
    help='Value loss coefficient.')
parser.add_argument(
    '--max_grad_norm',
    type=float,
    default=0.5,
    help='Max norm of student gradients.')
parser.add_argument(
    '--adv_max_grad_norm',
    type=float,
    default=0.5,
    help='Max norm of teacher gradients.')
parser.add_argument(
    '--normalize_returns',
    type=str2bool, nargs='?', const=True, default=False,
    help='Whether to normalize student returns.')
parser.add_argument(
    '--adv_normalize_returns',
    type=str2bool, nargs='?', const=True, default=False,
    help='Whether to normalize teacher returns.')
parser.add_argument(
    '--use_popart',
    type=str2bool, nargs='?', const=True, default=False,
    help='Whether to normalize student values via PopArt.')
parser.add_argument(
    '--adv_use_popart',
    type=str2bool, nargs='?', const=True, default=False,
    help='Whether to normalize teacher values using PopArt.')
parser.add_argument(
    '--seed', 
    type=int, 
    default=88, 
    help='Experiment random seed.')
parser.add_argument(
    '--num_processes',
    type=int,
    default=16,
    help='How many training CPU processes to use for experience collection. Note minimum of 2')
parser.add_argument(
    '--num_steps',
    type=int,
    default=256,
    help='Rollout horizon for A2C-style algorithms.')
parser.add_argument(
    '--ppo_epoch',
    type=int,
    default=5,
    help='Number of PPO epochs.')
parser.add_argument(
    '--adv_ppo_epoch',
    type=int,
    default=5,
    help='Number of PPO epochs used by teacher.')
parser.add_argument(
    '--num_mini_batch',
    type=int,
    default=1,
    help='Number of batches for PPO for student.')
parser.add_argument(
    '--adv_num_mini_batch',
    type=int,
    default=1,
    help='Number of batches for PPO for teacher.')
parser.add_argument(
    '--clip_param',
    type=float,
    default=0.2,
    help='PPO advantage clipping.')
parser.add_argument(
    '--clip_value_loss',
    type=str2bool,
    default=True,
    help='PPO value loss clipping.')
parser.add_argument(
    '--clip_reward',
    type=float,
    default=None,
    help="Amount to clip student rewards. By default no clipping.")
parser.add_argument(
    '--adv_clip_reward',
    type=float,
    default=None,
    help="Amount to clip teacher rewards. By default no clipping.")
parser.add_argument(
    '--num_env_steps',
    type=int,
    default=20000000,
    help='Number of environment steps for training.')

# Architecture arguments.
parser.add_argument(
    '--recurrent_arch',
    type=str,
    default='lstm',
    choices=['lstm'],
    help='RNN architecture for student and teacher.')
parser.add_argument(
    '--recurrent_agent',
    type=str2bool, nargs='?', const=True, default=True,
    help='Use a RNN architecture for student.')
parser.add_argument(
    '--recurrent_adversary_env',
    type=str2bool, nargs='?', const=True, default=False,
    help='Use a RNN architecture for teacher.')
parser.add_argument(
    '--recurrent_hidden_size',
    type=int,
    default=256,
    help='Recurrent hidden state size.')


# === UED arguments ===
parser.add_argument(
    '--ued_algo',
    type=str,
    default='domain_randomization',
    choices=['domain_randomization', 'minimax', 
             'paired', 'flexible_paired'],
    help='UED algorithm')
parser.add_argument(
    '--protagonist_plr',
    type=str2bool, nargs='?', const=True, default=False,
    help="PLR via protagonist's trajectories.")
parser.add_argument(
    '--antagonist_plr',
    type=str2bool, nargs='?', const=True, default=False,
    help="PLR via antagonist's lotrajectoriesss. If protagonist_plr is True, each agent trains using their own.")
parser.add_argument(
    '--use_reset_random_dr',
    type=str2bool, nargs='?', const=True, default=False,
    help='''
         Domain randomization (DR) resets using reset random. 
         If False, DR resets using a uniformly random adversary policy.
         Defaults to False for legacy reasons.''')


# PLR arguments.
parser.add_argument(
    "--use_plr",
    type=str2bool, nargs='?', const=True, default=True,
    help='Whether to use PLR.'
)
parser.add_argument(
    "--level_replay_strategy", 
    type=str,
    default='positive_value_loss',
    choices=['off', 'random', 'uniform', 'sequential',
            'policy_entropy', 'least_confidence', 'min_margin', 
            'gae', 'value_l1', 'signed_value_loss', 'positive_value_loss',
            'grounded_signed_value_loss', 'grounded_positive_value_loss',
            'one_step_td_error', 'alt_advantage_abs',
            'tscl_window'],
    help="PLR score function.")
parser.add_argument(
    "--level_replay_eps", 
    type=float,
    default=0.05,
    help="PLR epsilon for eps-greedy sampling. (Not typically used.)")
parser.add_argument(
    "--level_replay_score_transform",
    type=str, 
    default='rank', 
    choices=['constant', 'max', 'eps_greedy', 'rank', 'power', 'softmax', 'match', 'match_rank'], 
    help="PLR score transform.")
parser.add_argument(
    "--level_replay_temperature", 
    type=float,
    default=0.3,
    help="PLR replay distribution temperature.")
parser.add_argument(
    "--level_replay_schedule",
    type=str,
    default='proportionate',
    help="PLR schedule for annealing the replay rate.")
parser.add_argument(
    "--level_replay_rho",
    type=float, 
    default=0.5,
    help="Minimum fill ratio for PLR buffer before sampling replays.")
parser.add_argument(
    "--level_replay_prob", 
    type=float,
    default=0.8,
    help="Probability of sampling a replay level instead of a new level.")
parser.add_argument(
    "--level_replay_alpha",
    type=float, 
    default=1.0,
    help="PLR level score EWA smoothing factor.")
parser.add_argument(
    "--staleness_coef",
    type=float, 
    default=0.3,
    help="Staleness-sampling weighting.")
parser.add_argument(
    "--staleness_transform",
    type=str, 
    default='power',
    choices=['max', 'eps_greedy', 'rank', 'power', 'softmax'], 
    help="Staleness score transform.")
parser.add_argument(
    "--staleness_temperature",
    type=float, 
    default=1.0,
    help="Staleness distribution temperature.")
parser.add_argument(
    "--train_full_distribution",
    type=str2bool, nargs='?', const=True, default=True,
    help='Train on the full distribution of levels.')
parser.add_argument(
    "--level_replay_seed_buffer_size",
    type=int, 
    default=15,
    help="Size of PLR level buffer.")
parser.add_argument(
    "--level_replay_seed_buffer_priority",
    type=str, 
    default='replay_support',
    choices=['score', 'replay_support'], 
    help="How to prioritize level buffer members when capacity is reached.")
parser.add_argument(
    "--reject_unsolvable_seeds",
    type=str2bool, nargs='?', const=True, default=False,
    help='Do not add unsolvable seeds to the PLR buffer.')
parser.add_argument(
    "--no_exploratory_grad_updates",
    type=str2bool, nargs='?', const=True, default=True,
    help='Turns on Robust PLR: Only perform gradient updates for episodes on replay levels.'
)

# ACCEL arguments.
parser.add_argument(
    "--use_editor",
    type=str2bool, nargs='?', const=True, default=True,
    help='Turns on ACCEL: Evaluate mutated replay levels for entry in PLR buffer.')
parser.add_argument(
    "--level_editor_prob",
    type=float,
    default=1.0,
    help="Probability of mutating a replayed level under PLR.")
parser.add_argument(
    "--level_editor_method",
    type=str,
    default='random',
    choices=['random'],
    help="Method for mutating levels. ACCEL simply uses random mutations.")
parser.add_argument(
    "--base_levels",
    type=str,
    default='easy',
    choices=['batch', 'easy', 'hard'],
    help="What kind of replayed level under PLR do we edit?")
parser.add_argument(
    "--num_edits",
    type=int,
    default=1,
    help="Number of edits to make each time a level is mutated.")
parser.add_argument(
    "--use_accel_paired",
    type=str2bool, nargs='?', const=True, default=False,
    help='Whether to use paired regret estimate for level editor.')
parser.add_argument(
    "--accel_paired_score_function",
    type=str, default="paired", choices=["paired", "flex_paired"],
    help="Type of regret estimate for level editor"
)
parser.add_argument(
    "--use_lstm",
    type=str2bool, nargs='?', const=True, default=False,
    help='Whether to use LSTM architecture for BipedalWalker models.')

# BC arguments
parser.add_argument(
    '--use_behavioural_cloning',
    type=str2bool, nargs='?', const=True, default=False,
    help='Whether to use behavioural cloning')
parser.add_argument(
    '--kl_update_step',
    type=float,
    default=1,
    help='Number of steps after which KL loss should be used')
parser.add_argument(
    '--kl_loss_coef',
    type=float,
    default=0.0,
    help='KL divergence loss coefficient for behavioural cloning (default: 0.1)')
parser.add_argument(
    '--use_kl_only_agent',
    type=str2bool, nargs='?', const=True, default=False,
    help='Use behavioural cloning loss in agent only. Default behaviour is bc in both')

# Fine-tuning arguments.
parser.add_argument(
    '--xpid_finetune',
    default=None,
    help='Checkpoint directory containing model for fine-tuning.')
parser.add_argument(
    '--model_finetune',
    default='model',
    help='Name of .tar to load for fine-tuning.')

# Hardware arguments.
parser.add_argument(
    '--no_cuda',
    type=str2bool, nargs='?', const=True, default=False,
    help='Disables CUDA training.')

# Logging arguments.
parser.add_argument(
    '--xpid',
    default='latest',
    help='Name for the training run. Used for the name of the output results directory.')
parser.add_argument(
    '--log_dir',
    default='logs/accel/',
    help='Directory in which to save experimental outputs.')
parser.add_argument(
    '--log_interval',
    type=int,
    default=25,
    help='Log training stats every this many updates.')
parser.add_argument(
    "--checkpoint_interval", 
    type=int, 
    default=100,
    help="Save model every this many updates.")
parser.add_argument(
    "--archive_interval", 
    type=int, 
    default=5000,
    help="Save an archived checkpoint every this many updates.")
parser.add_argument(
    "--checkpoint_basis",
    type=str,
    default="student_grad_updates",
    choices=["num_updates", "student_grad_updates"],
    help=f'''Archive interval basis. 
             num_updates: By # update cycles (full rollout cycle across all agents); 
             student_grad_updates: By # grad updates performed by the student agent.''')
parser.add_argument(
    "--weight_log_interval", 
    type=int, 
    default=0,
    help="Save level weights every this many updates. *Only for PLR with a fixed level buffer.*")
parser.add_argument(
    '--render',
    type=str2bool, nargs='?', const=False, default=False,
    help='Render to environment to screen.')
parser.add_argument(
    "--checkpoint", 
    type=str2bool, nargs='?', const=True, default=True,
    help="Begin training from checkpoint. Needed for preemptible training on clusters.")
parser.add_argument(
    "--disable_checkpoint", 
    type=str2bool, nargs='?', const=True, default=False,
    help="Disable checkpointing.")
parser.add_argument(
    '--log_grad_norm',
    type=str2bool, nargs='?', const=True, default=False,
    help="Log the gradient norm of the actor-critic.")
parser.add_argument(
    '--log_action_complexity',
    type=str2bool, nargs='?', const=True, default=True,
    help="Log action-trajectory complexity metrics throughout training.")
parser.add_argument(
    '--log_replay_complexity',
    type=str2bool, nargs='?', const=True, default=True,
    help="Log complexity metrics of replay levels.")
parser.add_argument(
    '--log_plr_buffer_stats',
    type=str2bool, nargs='?', const=True, default=True,
    help="Log PLR buffer stats.")
parser.add_argument(
    "--verbose", 
    type=str2bool, nargs='?', const=True, default=False,
    help="Whether to print logs to stdout.")

# Evaluation arguments.
parser.add_argument(
    '--test_interval',
    type=int,
    default=250,
    help='Evaluate on test environments every this many updates.')
parser.add_argument(
    '--test_num_episodes',
    type=int,
    default=10,
    help='Number of test episodes per environment.')
parser.add_argument(
    '--test_num_processes',
    type=int,
    default=1,
    help='Number of test processes per environment.')
parser.add_argument(
    '--test_env_names',
    type=str,
    default="Cantilever-Adversarial-Mode1-v0,Cantilever-Adversarial-Mode2-v0,Cantilever-Adversarial-Mode3-v0,Cantilever-Adversarial-Mode4-v0,Cantilever-Adversarial-Mode5-v0,Cantilever-Adversarial-Mode12-v0,Cantilever-Adversarial-Mode23-v0,Cantilever-Adversarial-Mode34-v0,Cantilever-Adversarial-Mode45-v0,Cantilever-Adversarial-Mode123-v0,Cantilever-Adversarial-Mode234-v0,Cantilever-Adversarial-Mode345-v0,Cantilever-Adversarial-Mode1234-v0,Cantilever-Adversarial-Mode2345-v0,Cantilever-Adversarial-Mode12345-v0",
    help='CSV string of test environments for evaluation during training.')

# Environment arguments.
parser.add_argument(
    '--env_name',
    type=str,
    default='Cantilever-Adversarial-Main-v0',
    help='Environment to train on.')
parser.add_argument(
    '--handle_timelimits',
    type=str2bool, nargs='?', const=True, default=True,
    help="Bootstrap off of early termination states. Requires env to be wrapped by envs.wrappers.TimeLimit.")
parser.add_argument(
    '--singleton_env',
    type=str2bool, nargs='?', const=True, default=False,
    help="When using a fixed env, whether the same environment should also be reused across workers.")
parser.add_argument(
    '--use_global_critic',
    type=str2bool, nargs='?', const=True, default=False,
    help="Student's critic is fully observable. *Only for MultiGrid.*")
parser.add_argument(
    '--use_global_policy',
    type=str2bool, nargs='?', const=True, default=False,
    help="Student's policy is fully observable. *Only for MultiGrid.*")


# Cantilever-specific arguments.
parser.add_argument(
    '--sim_modes',
    type=list, default=[0,1,2,3,4],
    help="Cantilever environment number of modes/mode shape.")
parser.add_argument(
    '--num_sensors',
    type=int, default=5,
    help="Number of sensors.")
parser.add_argument(
    '--norm',
    type=bool, default=True,
    help="Normalisation of mode shape to unity.")
parser.add_argument(
    '--eps_length',
    type=int, default=200,
    help="length of environment episode.")
parser.add_argument(
		'--eff_ind_baseline',
		type=dict, default={
        'Cantilever-Adversarial-Mode1-v0': 1.1340411757648103,      # "(0,)"
        'Cantilever-Adversarial-Mode2-v0': 1.1478929855584956,      # "(1,)"
        'Cantilever-Adversarial-Mode3-v0': 3.812591429534031,       # "(2,)"
        'Cantilever-Adversarial-Mode4-v0': 1.1892742826499552,      # "(3,)"
        'Cantilever-Adversarial-Mode5-v0': 3.729404773008301,       # "(4,)"
        'Cantilever-Adversarial-Mode12-v0': 1.2426906498650618,     # "(0, 1)"
        'Cantilever-Adversarial-Mode23-v0': 3.5457205676537673,     # "(1, 2)"
        'Cantilever-Adversarial-Mode34-v0': 3.4720591209582685,     # "(2, 3)"
        'Cantilever-Adversarial-Mode45-v0': 1.3513950183631365,     # "(3, 4)"
        'Cantilever-Adversarial-Mode123-v0': 4.666560228475866,     # "(0, 1, 2)"
        'Cantilever-Adversarial-Mode234-v0': 4.724897476315033,     # "(1, 2, 3)"
        'Cantilever-Adversarial-Mode345-v0': 3.905513687874828,     # "(2, 3, 4)"
        'Cantilever-Adversarial-Mode1234-v0': 16.361524994321076,   # "(0, 1, 2, 3)"
        'Cantilever-Adversarial-Mode2345-v0': 4.323229720368495,    # "(1, 2, 3, 4)"
        'Cantilever-Adversarial-Mode12345-v0': 6.962692467362862,   # "(0, 1, 2, 3, 4)"
    	},
		help="Baseline for efficiency independence.")