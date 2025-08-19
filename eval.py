# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This file is a modified version from the code source
# https://github.com/facebookresearch/dcd

import sys
import os
import csv
import json
import argparse
import fnmatch
import re
from collections import defaultdict

import numpy as np
import torch
from baselines.common.vec_env import DummyVecEnv
from baselines.logger import HumanOutputFormat
from tqdm import tqdm

import os
import matplotlib as mpl
import matplotlib.pyplot as plt

from envs.registration import make as gym_make
from envs.wrappers import VecMonitor, ParallelAdversarialVecEnv, VecPreprocessImageWrapper,\
	FlattenMultiDiscreteActions
from util import DotDict, str2bool, make_agent, create_parallel_env, is_discrete_actions, envs_params
from arguments import parser

"""
Example usage:

python -m eval \
--env_name=Cantilever-Adversarial \
--xpid=<xpid> \
--base_path="~/logs/dcd" \
--result_path="eval_results/"
--verbose
"""
def parse_args():
	parser = argparse.ArgumentParser(description='Eval')

	parser.add_argument(
		'--base_path',
		type=str,
		default="logs/accel",
		help='Base path to experiment results directories.')
	parser.add_argument(
		'--xpid',
		type=str,
		default='latest',
		help='Experiment ID (result directory name) for evaluation.')
	parser.add_argument(
		'--prefix',
		type=str,
		default=None,
		help='Experiment ID prefix for evaluation (evaluate all matches).'
	)
	parser.add_argument(
		'--env_names',
		type=str,
		default='Cantilever-Adversarial-Main-v0',
		help='CSV string of evaluation environments.')
	parser.add_argument(
		'--result_path',
		type=str,
		default='eval_results/',
		help='Relative path to evaluation results directory.')
	parser.add_argument(
		'--accumulator',
		type=str,
		default=None,
		help="Function for accumulating across multiple evaluation runs.")
	parser.add_argument(
		'--singleton_env',
		type=str2bool, nargs='?', const=True, default=False,
		help="When using a fixed env, whether the same environment should also be reused across workers.")
	parser.add_argument(
		'--seed', 
		type=int, 
		default=1, 
		help='Random seed.')
	parser.add_argument(
		'--max_seeds', 
		type=int, 
		default=None, 
		help='Maximum number of matched experiment IDs to evaluate.')
	parser.add_argument(
		'--num_processes',
		type=int,
		default=1,
		help='Number of CPU processes to use.')
	parser.add_argument(
		'--max_num_processes',
		type=int,
		default=10,
		help='Maximum number of CPU processes to use.')
	parser.add_argument(
		'--num_episodes',
		type=int,
		default=100,
		help='Number of evaluation episodes per xpid per environment.')
	parser.add_argument(
		'--model_tar',
		type=str,
		default='model',
		help='Name of .tar to evaluate.')
	parser.add_argument(
		'--model_name',
		type=str,
		default='agent',
		choices=['agent', 'adversary_agent'],
		help='Which agent to evaluate.')
	parser.add_argument(
		'--deterministic',
		type=str2bool, nargs='?', const=True, default=False,
		help="Evaluate policy greedily.")
	parser.add_argument(
		'--verbose',
		type=str2bool, nargs='?', const=True, default=True,
		help="Show logging messages in stdout")
	parser.add_argument(
		'--render',
		type=str2bool, nargs='?', const=True, default=False,
		help="Render environment in first evaluation process to screen.")
	parser.add_argument(
		'--record_video',
		type=str2bool, nargs='?', const=True, default=False,
		help="Record video of first environment evaluation process.")
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
		help="Length of environment episode.")
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

	return parser.parse_args()


class Evaluator(object):
	def __init__(self, 
		env_names, 
		num_processes, 
		num_episodes=10, 
		record_video=False,
		solve_baseline = 0, 
		device='cpu', 
		**kwargs):
		self.kwargs = kwargs # kwargs for env wrappers
		self._init_parallel_envs(
			env_names, num_processes, device=device, record_video=record_video, **kwargs)
		self.num_episodes = num_episodes
		self.solved_threshold = solve_baseline

	def get_stats_keys(self):
		keys = []
		for env_name in self.env_names:
			keys += [f'solved_rate:{env_name}', f'test_returns:{env_name}']
		return keys

	@staticmethod
	def make_env(env_name, record_video=False, **kwargs):
		env = gym_make(env_name, **kwargs)
		env = FlattenMultiDiscreteActions(env)
		return env

	@staticmethod
	def wrap_venv(venv, env_name, device='cpu'):
		venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
		venv = VecPreprocessImageWrapper(venv=venv)
		return venv

	def _init_parallel_envs(self, env_names, num_processes, device=None, record_video=False, **kwargs):
		self.env_names = env_names
		self.num_processes = num_processes
		self.device = device
		self.venv = {env_name:None for env_name in env_names}

		make_fn = []
		for env_name in env_names:
			make_fn = [lambda: Evaluator.make_env(env_name, record_video, **kwargs)]*self.num_processes
			venv = ParallelAdversarialVecEnv(make_fn, adversary=False, is_eval=True)
			venv = Evaluator.wrap_venv(venv, env_name, device=device)
			self.venv[env_name] = venv

		self.is_discrete_actions = is_discrete_actions(self.venv[env_names[0]])

	def close(self):
		for _, venv in self.venv.items():
			venv.close()

	def evaluate(self, 
		agent, 
		deterministic=False, 
		show_progress=True,
		render=False,
		accumulator='mean'):

		# Evaluate agent for N episodes
		venv = self.venv
		env_returns = {}
		env_solved_episodes = {}
		env_reward_metric = {}
		env_node_Id = {}
		
		for env_name, venv in self.venv.items():
			returns = []
			reward_metric = []
			node_Id = []
			solved_episodes = 0

			obs = venv.reset()
			recurrent_hidden_states = torch.zeros(
				self.num_processes, agent.algo.actor_critic.recurrent_hidden_state_size, device=self.device)
			if agent.algo.actor_critic.is_recurrent and agent.algo.actor_critic.rnn.arch == 'lstm':
				recurrent_hidden_states = (recurrent_hidden_states, torch.zeros_like(recurrent_hidden_states))
			masks = torch.ones(self.num_processes, 1, device=self.device)

			# Initialize progress bar
			pbar = None
			if show_progress:
				pbar = tqdm(total=self.num_episodes, desc=f"Evaluating {env_name}", unit="episode")

			while len(returns) < self.num_episodes:
				# Sample actions
				with torch.no_grad():
					_, action, _, recurrent_hidden_states = agent.act(
						obs, recurrent_hidden_states, masks, deterministic=deterministic)

				# Observe reward and next obs
				action = action.cpu().numpy()[0]
				if not self.is_discrete_actions:
					action = agent.process_action(action)
				obs, reward, done, infos = venv.step(action)
				masks = torch.tensor(
					[[0.0] if done_ else [1.0] for done_ in done],
					dtype=torch.float32,
					device=self.device)

				for i, info in enumerate(infos):
					if 'episode' in info.keys():
						returns.append(info['episode']['r'])
						reward_metric.append(info['reward_metric'])
						node_Id.append(info['node_Id'])
						if reward_metric[-1] > self.solved_threshold[str(env_name)]:
							solved_episodes += 1
						if pbar:
							pbar.set_postfix(reward_metric = reward_metric[-1],
											avg_return=np.mean(returns), 
											solved_rate=solved_episodes / len(returns))
							pbar.update(1)

						# zero hidden states
						if agent.is_recurrent:
							recurrent_hidden_states[0][i].zero_()
							recurrent_hidden_states[1][i].zero_()

						if len(returns) >= self.num_episodes:
							break

				if render:
					venv.render_to_screen()

			if pbar:
				pbar.close()	

			env_returns[env_name] = returns
			env_solved_episodes[env_name] = solved_episodes
			env_reward_metric[env_name] = reward_metric
			env_node_Id[env_name] = node_Id


		stats = {}
		for env_name in self.env_names:
			if accumulator == 'mean':
				stats[f"solved_rate:{env_name}"] = env_solved_episodes[env_name]/self.num_episodes

			if accumulator == 'mean':
				stats[f"test_returns:{env_name}"] = np.mean(env_returns[env_name])
				stats[f"reward_metric:{env_name}"] = np.mean(env_reward_metric[env_name])
			else:
				stats[f"solved_rate:{env_name}"] = env_solved_episodes[env_name]
				stats[f"test_returns:{env_name}"] = env_returns[env_name]
				stats[f"reward_metric:{env_name}"] = env_reward_metric[env_name]
			stats[f"node_Id:{env_name}"] = env_node_Id[env_name][-1]
		return stats



def _get_zs_cantilever_env_names():
    env_names = [
        'Cantilever-Adversarial-Mode1-v0',
        'Cantilever-Adversarial-Mode2-v0',
        'Cantilever-Adversarial-Mode3-v0',
        'Cantilever-Adversarial-Mode4-v0',
        'Cantilever-Adversarial-Mode5-v0',
        'Cantilever-Adversarial-Mode12-v0',
        'Cantilever-Adversarial-Mode23-v0',
        'Cantilever-Adversarial-Mode34-v0',
        'Cantilever-Adversarial-Mode45-v0',
        'Cantilever-Adversarial-Mode123-v0',
        'Cantilever-Adversarial-Mode234-v0',
        'Cantilever-Adversarial-Mode345-v0',
        'Cantilever-Adversarial-Mode1234-v0',
        'Cantilever-Adversarial-Mode2345-v0',
        'Cantilever-Adversarial-Mode12345-v0',
    ]
    return env_names


if __name__ == '__main__':
	os.environ["OMP_NUM_THREADS"] = "1"

	args = DotDict(vars(parse_args()))
	args.num_processes = min(args.num_processes, args.num_episodes)

	#=== Initialise environment ====
	env_kwargs = {
    'sim_modes': args.sim_modes,
    'num_sensors': args.num_sensors,
    'render': args.render,
    'norm': args.norm,
    }

	pyansys_env = envs_params(env_kwargs)
	args.pyansys_env = pyansys_env.get_data()
	pyansys_env.close()

	# === Determine device ====
	device = 'cpu'

	# === Load checkpoint ===
	# Load meta.json into flags object
	cwd = os.getcwd()
	base_path = os.path.join(cwd, args.base_path)
	#base_path = os.path.expandvars(os.path.expanduser(args.base_path))

	xpids = [args.xpid]
	if args.prefix is not None:
		all_xpids = fnmatch.filter(os.listdir(base_path), f"{args.prefix}*")
		filter_re = re.compile('.*_[0-9]*$')
		xpids = [x for x in all_xpids if filter_re.match(x)]

	# Set up results management
	os.makedirs(args.result_path, exist_ok=True)
	if args.prefix is not None:
		result_fname = args.prefix
	else:
		result_fname = args.xpid
	result_fname = f"{result_fname}-{args.model_tar}-{args.model_name}"
	result_fpath = os.path.join(args.result_path, result_fname)
	if os.path.exists(f'{result_fpath}.csv'):
		result_fpath = os.path.join(args.result_path, f'{result_fname}_redo')
	result_fpath = f'{result_fpath}.csv'

	csvout = open(result_fpath, 'w', newline='')
	csvwriter = csv.writer(csvout)

	env_results = defaultdict(list)

	# Get envs

	env_names = _get_zs_cantilever_env_names()
	
	num_envs = len(env_names)
	if num_envs*args.num_processes > args.max_num_processes:
		chunk_size = args.max_num_processes//args.num_processes
	else:
		chunk_size = num_envs

	num_chunks = int(np.ceil(num_envs/chunk_size))

	if args.record_video:
		num_chunks = 1
		chunk_size = 1
		args.num_processes = 1

	num_seeds = 0
	for xpid in xpids:
		if args.max_seeds is not None and num_seeds >= args.max_seeds:
			break

		xpid_dir = os.path.join(base_path, xpid)
		meta_json_path = os.path.join(xpid_dir, 'meta.json')

		model_tar = f'{args.model_tar}.tar'
		checkpoint_path = os.path.join(xpid_dir, model_tar)

		if os.path.exists(checkpoint_path):
			meta_json_file = open(meta_json_path)       
			xpid_flags = DotDict(json.load(meta_json_file)['args'])

			make_fn = [lambda: Evaluator.make_env(env_names[0], 
										 			sim_modes = args.sim_modes,
           											num_sensors = args.num_sensors,
            										pyansys_env = args.pyansys_env)]
			dummy_venv = ParallelAdversarialVecEnv(make_fn, adversary=False, is_eval=True)
			dummy_venv = Evaluator.wrap_venv(dummy_venv, env_name=env_names[0], device=device)

			# Load the agent
			agent = make_agent(name='agent', env=dummy_venv, args=xpid_flags, device=device)

			try:
				checkpoint = torch.load(checkpoint_path, map_location='cpu')
			except:
				continue
			model_name = args.model_name

			if 'runner_state_dict' in checkpoint:
				agent.algo.actor_critic.load_state_dict(checkpoint['runner_state_dict']['agent_state_dict'][model_name])
			else:
				agent.algo.actor_critic.load_state_dict(checkpoint)

			num_seeds += 1

			# Evaluate environment batch in increments of chunk size
			for i in range(num_chunks):
				start_idx = i*chunk_size
				env_names_ = env_names[start_idx:start_idx+chunk_size]

				# Evaluate the model
				xpid_flags.update(args)
				xpid_flags.update({"use_skip": False})

				evaluator = Evaluator(env_names_, 
					num_processes=args.num_processes, 
					num_episodes=args.num_episodes, 
					record_video=args.record_video,
					sim_modes = args.sim_modes,
					solve_baseline = args.eff_ind_baseline,
           			num_sensors = args.num_sensors,
            		pyansys_env = args.pyansys_env)

				stats = evaluator.evaluate(agent, 
					deterministic=args.deterministic, 
					show_progress=args.verbose,
					render=args.render,
					accumulator=args.accumulator)
				for k,v in stats.items():
					if args.accumulator:
						env_results[k].append(v)
					else:
						if k.startswith("node_Id") or k.startswith("solved_rate"):
							env_results[k] = v
						else:
							env_results[k] += v

				evaluator.close()
		else:
			print(f'No model path {checkpoint_path}')

	output_results = {}
	for k,_ in stats.items():
		results = env_results[k]
		if k.startswith("node_Id"):
			output_results[k] = f'{results} +/- {None}'
		else:
			output_results[k] = f'{np.mean(results):.2f} +/- {np.std(results):.2f}'
			q1 = np.percentile(results, 25, interpolation='midpoint')
			q3 = np.percentile(results, 75, interpolation='midpoint')
			median = np.median(results)
			output_results[f'iq_{k}'] = f'{q1:.2f}--{median:.2f}--{q3:.2f}'			
		print(f"{k}: {output_results[k]}")
	HumanOutputFormat(sys.stdout).writekvs(output_results)

	if args.accumulator:
		csvwriter.writerow(['metric',] + [x for x in range(num_seeds)])
	else:
		csvwriter.writerow(['metric',] + [x for x in range(num_seeds*args.num_episodes)])
	for k,v in env_results.items():
		if k.startswith("node_Id"):
			row = [k,] + list(v)
		elif k.startswith("solved_rate"):
			row = [k,] + [v]
		else:
			row = [k,] + v
		csvwriter.writerow(row)
