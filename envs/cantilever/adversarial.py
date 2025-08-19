"""An adversarial environment for the Cantilever problem.
This environment allows an adversary to select different modes combinations of the cantilever
beam, which can be used to test the robustness of an agent's policy.
It is designed to be used with reinforcement learning algorithms, where the agent
learns to optimise sensor locations on the cantilever beam while the adversary can change the environment
to make the task more difficult.
This environment is a subclass of CantileverEnv_v0_1 and extends its functionality to include adversarial actions.
"""
import gym
import numpy as np
from . import cantilever_env
from . import register
import logging
#Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def DEFAULT_LEVEL_PARAMS_VEC(num_mode, seed = 0, per_split = 0.75):
    np.random.seed(seed)
    n = len(num_mode)
    mode_shape_comb = [
        tuple(num_mode[i:i+l])
        for l in range(1, n+1)          # Length of subsequence (1 → n)
        for i in range(n - l + 1)       # Starting index for this length
    ]
    training_env = np.random.choice(len(mode_shape_comb), 
                                      int(per_split*len(mode_shape_comb)),
                                      replace = False)
    return {mode_shape_comb[idx]: level for idx, level 
            in zip(np.sort(training_env), np.arange(len(training_env)))}

class CantileverAdversarial(cantilever_env.CantileverEnv_v0_1):
    def __init__(self, env_config, random_z_dim=10):
        super().__init__(config=env_config)
        self.passable = True
        # Initialize other attributes
        self.pyansys_env = env_config.get("pyansys_env")
        self.level_dict = DEFAULT_LEVEL_PARAMS_VEC(env_config['sim_modes'])
        print(f"Level dict: {self.level_dict}")
        self.adversary_max_steps = 1 #Adversary max steps for selection of the on modeshape combination
        self.level_params_vec = list(self.level_dict.keys())
        self.random_z_dim = random_z_dim
        #Pyansys data  
        self.phi_data = self.pyansys_env["phi_data"]
        self.correlation_covariance_matrix_data = self.pyansys_env["correlation_covariance_matrix_data"]
        self.encode_data = self.pyansys_env["encode_data"] 
        self.phi_index = env_config.get('phi_index', None)
        
        # Create spaces for adversary agent's specs.
        self.adversary_action_dim = len(self.level_params_vec)
        self.adversary_action_space = gym.spaces.Discrete(self.adversary_action_dim)
        self.adversary_randomz_obs_space = gym.spaces.Box(
            low=0, high=1.0, shape=(random_z_dim,), dtype=np.float32)
        num_nodes = 1462
        self.adversary_obs_space = gym.spaces.Box(low=0, high=1, shape=(num_nodes + len(env_config['sim_modes']),), 
                                                  dtype=np.int8)
        self.adversary_ts_obs_space = gym.spaces.Box(low=0, high=self.adversary_max_steps,
                                                      shape=(1,), dtype="uint8")
        #Adversary observation
        self.adversary_observation_space = gym.spaces.Dict(
            {'node_binary': self.adversary_obs_space,
             'time_step': self.adversary_ts_obs_space,
             'random_z': self.adversary_randomz_obs_space
             })
        
    def reset_agent(self):
        try:
            """Resets the agent's start position, but leaves structural condition unchanged.
            Reset the current environment level, i.e. only the student agent's state. 
            Do not change the actual environment configuration. 
            Returns the first observation in a new episode starting in that level"""
            obs, info = super()._reset_env()
            return obs
        except Exception as e:
            logging.error(f"Error in reset_agent: {e}")
            raise

    @property
    def processed_action_dim(self):
        return 1
    
    def reset_random(self):
         """Reset the environment to a random level, and return the observation 
            for the first time step in this level.
            For Cantilever: Reset to a random combination of the modeshape
         """
         try:
            self.phi_index = np.random.choice(self.level_params_vec)
            return self.reset_agent()
         except Exception as e:
            logging.error(f"Error in reset_random: {e}")
            raise
     
    def reset(self):
        """Reset the environment to an initial state. 
           For example, for a maze environment, this initial state 
           is an empty grid.
        """
        try:
            self.adversary_step_count = 0
            self.phi_index = self.level_params_vec[0]
            node_binary = self.reset_agent()
            obs = {
                'node_binary': node_binary,
                'time_step' : [self.adversary_step_count],
                'random_z': self.generate_random_z()
            }
            return obs
        except Exception as e:
            logging.error(f"Error in reset: {e}")
            raise

    def generate_random_z(self):
        try:
            return np.random.uniform(size=(self.random_z_dim,)).astype(np.float32)
        except Exception as e:
            logging.error(f"Error in generate_random_z: {e}")
            raise

    def step_adversary(self, action):
        """The adversary action is to change the location of the point mass
            in the grid
            Args:
                node_index : index of node id
        """
        try:
            self.phi_index = self.level_params_vec[action]
            node_binary = self.reset_agent()
            done = True
            self.adversary_step_count += 1
            obs = {
                'node_binary': node_binary,
                'time_step' : [self.adversary_step_count],
                'random_z': self.generate_random_z()
                }
            return  obs, 0, done, {}
        except Exception as e:
            logging.error(f"Error in step_adversary: {e}")
            raise

    def reset_to_level(self, level):
        """Level is equivalent to node_index
        """
        try:
            if isinstance(level, str):
                action = int(level)
            else:
                action = level
            obs, _, done, _ = self.step_adversary(action)
            obs = self.reset_agent()
            return obs
        except Exception as e:
            logging.error(f"Error in reset_to_level: {e}")
            raise
      
    def mutate_level(self, num_edits=0):
        """Given the environment construct
        A mutation is therefore num_edits reposition of
        the start location of the sensors but still
        in the same environment level.
        """
        try:
            if num_edits > 0:
                if num_edits <= self.num_sensors:
                    edit_sensors = np.random.choice(self.num_sensors, num_edits, replace=False)
                    edit_sensor_ini_loc =  np.random.choice(self.mutation_init_space, 
                                                        num_edits,
                                                        replace=False)
                else:
                    raise ValueError('Number of mutation edit larger than number of sensors')
                #Reset env to the current level
            node_binary, info = super()._reset_env(edit_sensors = edit_sensors,
                    edit_sensor_ini_loc = edit_sensor_ini_loc)
            
            self.adversary_step_count = 0
            return node_binary
        except Exception as e:
            logging.error(f"Error in mutate_level: {e}")
            raise
      
    def get_complexity_info(self):
        #* Return the complexity information of the environment.
        try:
            complexity_info = { 
                'level' : self.level_dict[self.phi_index]
                }       
            return complexity_info
        except Exception as e:
            logging.error(f"Error in get_complexity_info: {e}")
            raise
        
    @property
    def level(self):
        """Return the current level of the environment.
        """
        return self.level_dict[self.phi_index]
          
##TODO try to remove the env_config and pass directly from __init__ to super.__init__

class AdversarialEnvMain(CantileverAdversarial):
    def __init__(self,
                sim_modes,
                num_sensors,
                pyansys_env, 
                 ):
        env_config = {
            "sim_modes": sim_modes,
            "num_sensors": num_sensors,
            "pyansys_env": pyansys_env,
            "phi_index" : (1,),
        }
        super().__init__(env_config=env_config)
      
class AdversarialEnv1(CantileverAdversarial):
    def __init__(self,
                sim_modes,
                num_sensors,
                pyansys_env, 
                 ):
        env_config = {
            "sim_modes": sim_modes,
            "num_sensors": num_sensors,
            "pyansys_env": pyansys_env,
            "phi_index" : (0,),
        }
        super().__init__(env_config=env_config)
    def reset(self):
        obs, info = super()._reset_env()
        return obs

class AdversarialEnv2(CantileverAdversarial):
    def __init__(self,
                sim_modes,
                num_sensors,
                pyansys_env, 
                 ):
        env_config = {
            "sim_modes": sim_modes,
            "num_sensors": num_sensors,
            "pyansys_env": pyansys_env,
            "phi_index" : (1,),
        }
        super().__init__(env_config=env_config)
    def reset(self):
        obs, info = super()._reset_env()
        return obs

class AdversarialEnv3(CantileverAdversarial):
    def __init__(self,
                sim_modes,
                num_sensors,
                pyansys_env, 
                 ):
        env_config = {
            "sim_modes": sim_modes,
            "num_sensors": num_sensors,
            "pyansys_env": pyansys_env,
            "phi_index" : (2,),
        }
        super().__init__(env_config=env_config)
    def reset(self):
        obs, info = super()._reset_env()
        return obs

class AdversarialEnv4(CantileverAdversarial):
    def __init__(self,
                sim_modes,
                num_sensors,
                pyansys_env, 
                 ):
        env_config = {
            "sim_modes": sim_modes,
            "num_sensors": num_sensors,
            "pyansys_env": pyansys_env,
            "phi_index" : (3,),
        }
        super().__init__(env_config=env_config)
    def reset(self):
        obs, info = super()._reset_env()
        return obs
    
class AdversarialEnv5(CantileverAdversarial):
    def __init__(self,
                sim_modes,
                num_sensors,
                pyansys_env, 
                 ):
        env_config = {
            "sim_modes": sim_modes,
            "num_sensors": num_sensors,
            "pyansys_env": pyansys_env,
            "phi_index" : (4,),
        }
        super().__init__(env_config=env_config)
    def reset(self):
        obs, info = super()._reset_env()
        return obs

class AdversarialEnv6(CantileverAdversarial):
    def __init__(self,
                sim_modes,
                num_sensors,
                pyansys_env, 
                 ):
        env_config = {
            "sim_modes": sim_modes,
            "num_sensors": num_sensors,
            "pyansys_env": pyansys_env,
            "phi_index" : (5,),
        }
        super().__init__(env_config=env_config)
    def reset(self):
        obs, info = super()._reset_env()
        return obs



class AdversarialEnv12(CantileverAdversarial):
    def __init__(self,
                sim_modes,
                num_sensors,
                pyansys_env, 
                 ):
        env_config = {
            "sim_modes": sim_modes,
            "num_sensors": num_sensors,
            "pyansys_env": pyansys_env,
            "phi_index" : (0,1,),
        }
        super().__init__(env_config=env_config)
    def reset(self):
        obs, info = super()._reset_env()
        return obs

class AdversarialEnv23(CantileverAdversarial):
    def __init__(self,
                sim_modes,
                num_sensors,
                pyansys_env, 
                 ):
        env_config = {
            "sim_modes": sim_modes,
            "num_sensors": num_sensors,
            "pyansys_env": pyansys_env,
            "phi_index" : (1,2,),
        }
        super().__init__(env_config=env_config)
    def reset(self):
        obs, info = super()._reset_env()
        return obs
    
class AdversarialEnv34(CantileverAdversarial):
    def __init__(self,
                sim_modes,
                num_sensors,
                pyansys_env, 
                 ):
        env_config = {
            "sim_modes": sim_modes,
            "num_sensors": num_sensors,
            "pyansys_env": pyansys_env,
            "phi_index" : (2,3,),
        }
        super().__init__(env_config=env_config)
    def reset(self):
        obs, info = super()._reset_env()
        return obs
    
class AdversarialEnv45(CantileverAdversarial):
    def __init__(self,
                sim_modes,
                num_sensors,
                pyansys_env, 
                 ):
        env_config = {
            "sim_modes": sim_modes,
            "num_sensors": num_sensors,
            "pyansys_env": pyansys_env,
            "phi_index" : (3,4,),
        }
        super().__init__(env_config=env_config)
    def reset(self):
        obs, info = super()._reset_env()
        return obs
    
class AdversarialEnv56(CantileverAdversarial):
    def __init__(self,
                sim_modes,
                num_sensors,
                pyansys_env, 
                 ):
        env_config = {
            "sim_modes": sim_modes,
            "num_sensors": num_sensors,
            "pyansys_env": pyansys_env,
            "phi_index" : (4,5,),
        }
        super().__init__(env_config=env_config)
    def reset(self):
        obs, info = super()._reset_env()
        return obs

class AdversarialEnv123(CantileverAdversarial):
    def __init__(self,
                sim_modes,
                num_sensors,
                pyansys_env, 
                 ):
        env_config = {
            "sim_modes": sim_modes,
            "num_sensors": num_sensors,
            "pyansys_env": pyansys_env,
            "phi_index" : (0,1,2,),
        }
        super().__init__(env_config=env_config)
    def reset(self):
        obs, info = super()._reset_env()
        return obs

class AdversarialEnv234(CantileverAdversarial):
    def __init__(self,
                sim_modes,
                num_sensors,
                pyansys_env, 
                 ):
        env_config = {
            "sim_modes": sim_modes,
            "num_sensors": num_sensors,
            "pyansys_env": pyansys_env,
            "phi_index" : (1,2,3,),
        }
        super().__init__(env_config=env_config)
    def reset(self):
        obs, info = super()._reset_env()
        return obs

class AdversarialEnv345(CantileverAdversarial):
    def __init__(self,
                sim_modes,
                num_sensors,
                pyansys_env, 
                 ):
        env_config = {
            "sim_modes": sim_modes,
            "num_sensors": num_sensors,
            "pyansys_env": pyansys_env,
            "phi_index" : (2,3,4),
        }
        super().__init__(env_config=env_config)
    def reset(self):
        obs, info = super()._reset_env()
        return obs
    
class AdversarialEnv456(CantileverAdversarial):
    def __init__(self,
                sim_modes,
                num_sensors,
                pyansys_env, 
                 ):
        env_config = {
            "sim_modes": sim_modes,
            "num_sensors": num_sensors,
            "pyansys_env": pyansys_env,
            "phi_index" : (3,4,5,),
        }
        super().__init__(env_config=env_config)
    def reset(self):
        obs, info = super()._reset_env()
        return obs

class AdversarialEnv1234(CantileverAdversarial):
    def __init__(self,
                sim_modes,
                num_sensors,
                pyansys_env, 
                 ):
        env_config = {
            "sim_modes": sim_modes,
            "num_sensors": num_sensors,
            "pyansys_env": pyansys_env,
            "phi_index" : (0,1,2,3,),
        }
        super().__init__(env_config=env_config)
    def reset(self):
        obs, info = super()._reset_env()
        return obs
    
class AdversarialEnv2345(CantileverAdversarial):
    def __init__(self,
                sim_modes,
                num_sensors,
                pyansys_env, 
                 ):
        env_config = {
            "sim_modes": sim_modes,
            "num_sensors": num_sensors,
            "pyansys_env": pyansys_env,
            "phi_index" : (1,2,3,4,),
        }
        super().__init__(env_config=env_config)
    def reset(self):
        obs, info = super()._reset_env()
        return obs
    
class AdversarialEnv3456(CantileverAdversarial):
    def __init__(self,
                sim_modes,
                num_sensors,
                pyansys_env, 
                 ):
        env_config = {
            "sim_modes": sim_modes,
            "num_sensors": num_sensors,
            "pyansys_env": pyansys_env,
            "phi_index" : (2,3,4,5,),
        }
        super().__init__(env_config=env_config)
    def reset(self):
        obs, info = super()._reset_env()
        return obs
    
class AdversarialEnv12345(CantileverAdversarial):
    def __init__(self,
                sim_modes,
                num_sensors,
                pyansys_env, 
                 ):
        env_config = {
            "sim_modes": sim_modes,
            "num_sensors": num_sensors,
            "pyansys_env": pyansys_env,
            "phi_index" : (0,1,2,3,4,),
        }
        super().__init__(env_config=env_config)
    def reset(self):
        obs, info = super()._reset_env()
        return obs
    
class AdversarialEnv23456(CantileverAdversarial):
    def __init__(self,
                sim_modes,
                num_sensors,
                pyansys_env, 
                 ):
        env_config = {
            "sim_modes": sim_modes,
            "num_sensors": num_sensors,
            "pyansys_env": pyansys_env,
            "phi_index" : (1,2,3,4,5,),
        }
        super().__init__(env_config=env_config)
    def reset(self):
        obs, info = super()._reset_env()
        return obs
    
class AdversarialEnv123456(CantileverAdversarial):
    def __init__(self,
                sim_modes,
                num_sensors,
                pyansys_env, 
                 ):
        env_config = {
            "sim_modes": sim_modes,
            "num_sensors": num_sensors,
            "pyansys_env": pyansys_env,
            "phi_index" : (0,1,2,3,4,5,),
        }
        super().__init__(env_config=env_config)
    def reset(self):
        obs, info = super()._reset_env()
        return obs

if hasattr(__loader__, 'name'):
  module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
  module_path = __loader__.fullname

# Registering all environments based on the provided dictionary.
register.register(
    env_id='Cantilever-Adversarial-Main-v0',
    entry_point=module_path + ':AdversarialEnvMain',
    max_episode_steps=200,
)

register.register(
    env_id='Cantilever-Adversarial-Mode1-v0',
    entry_point=module_path + ':AdversarialEnv1',
    max_episode_steps=200,
)

register.register(
    env_id='Cantilever-Adversarial-Mode2-v0',
    entry_point=module_path + ':AdversarialEnv2',
    max_episode_steps=200,
)

register.register(
    env_id='Cantilever-Adversarial-Mode3-v0',
    entry_point=module_path + ':AdversarialEnv3',
    max_episode_steps=200,
)

register.register(
    env_id='Cantilever-Adversarial-Mode4-v0',
    entry_point=module_path + ':AdversarialEnv4',
    max_episode_steps=200,
)

register.register(
    env_id='Cantilever-Adversarial-Mode5-v0',
    entry_point=module_path + ':AdversarialEnv5',
    max_episode_steps=200,
)

register.register(
    env_id='Cantilever-Adversarial-Mode6-v0',
    entry_point=module_path + ':AdversarialEnv6',
    max_episode_steps=200,
)

register.register(
    env_id='Cantilever-Adversarial-Mode12-v0',
    entry_point=module_path + ':AdversarialEnv12',
    max_episode_steps=200,
)

register.register(
    env_id='Cantilever-Adversarial-Mode23-v0',
    entry_point=module_path + ':AdversarialEnv23',
    max_episode_steps=200,
)

register.register(
    env_id='Cantilever-Adversarial-Mode34-v0',
    entry_point=module_path + ':AdversarialEnv34',
    max_episode_steps=200,
)

register.register(
    env_id='Cantilever-Adversarial-Mode45-v0',
    entry_point=module_path + ':AdversarialEnv45',
    max_episode_steps=200,
)

register.register(
    env_id='Cantilever-Adversarial-Mode56-v0',
    entry_point=module_path + ':AdversarialEnv56',
    max_episode_steps=200,
)

register.register(
    env_id='Cantilever-Adversarial-Mode123-v0',
    entry_point=module_path + ':AdversarialEnv123',
    max_episode_steps=200,
)

register.register(
    env_id='Cantilever-Adversarial-Mode234-v0',
    entry_point=module_path + ':AdversarialEnv234',
    max_episode_steps=200,
)

register.register(
    env_id='Cantilever-Adversarial-Mode345-v0',
    entry_point=module_path + ':AdversarialEnv345',
    max_episode_steps=200,
)

register.register(
    env_id='Cantilever-Adversarial-Mode456-v0',
    entry_point=module_path + ':AdversarialEnv456',
    max_episode_steps=200,
)

register.register(
    env_id='Cantilever-Adversarial-Mode1234-v0',
    entry_point=module_path + ':AdversarialEnv1234',
    max_episode_steps=200,
)

register.register(
    env_id='Cantilever-Adversarial-Mode2345-v0',
    entry_point=module_path + ':AdversarialEnv2345',
    max_episode_steps=200,
)

register.register(
    env_id='Cantilever-Adversarial-Mode3456-v0',
    entry_point=module_path + ':AdversarialEnv3456',
    max_episode_steps=200,
)

register.register(
    env_id='Cantilever-Adversarial-Mode12345-v0',
    entry_point=module_path + ':AdversarialEnv12345',
    max_episode_steps=200,
)

register.register(
    env_id='Cantilever-Adversarial-Mode23456-v0',
    entry_point=module_path + ':AdversarialEnv23456',
    max_episode_steps=200,
)

register.register(
    env_id='Cantilever-Adversarial-Mode123456-v0',
    entry_point=module_path + ':AdversarialEnv123456',
    max_episode_steps=200,
)