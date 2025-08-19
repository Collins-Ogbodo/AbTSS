from util.eff_indep import effective_independence
from util import envs_params
from arguments import parser
from envs.cantilever import *
import os
import csv
import numpy as np
from scipy.linalg import solve_triangular
import logging

def _info_metric(state: np.ndarray, 
                 observation_space_node: np.ndarray,
                 correlation_covariance_matrix,
                 phi) -> float:
    """
    Calculates the information entropy index for the sensor placements.
    
    The reward function is an information theoritic metric which measures the information gain for each configuration
        References: 
        [1] Zhang, Jie, et al. "Optimal sensor placement for multi-setup modal analysis of structures." 
            Journal of Sound and Vibration 401 (2017): 214-232.
        [2] Papadimitriou, Costas, and Geert Lombaert. "The effect of prediction error correlation on optimal sensor placement in structural dynamics." 
            Mechanical Systems and Signal Processing 28 (2012): 105-127.
        [3] Wang, Ying, et al. "Advancements in Optimal Sensor Placement for Enhanced Structural Health Monitoring: Current Insights and Future Prospects." 
            Buildings 13.12 (2023): 3129.
        [4] Wang, Zhi, Han-Xiong Li, and Chunlin Chen. "Reinforcement learning-based optimal sensor placement for spatiotemporal modeling." 
            IEEE transactions on cybernetics 50.6 (2019): 2861-2871.
        [5] Kammer, Daniel C. "Sensor placement for on-orbit modal identification and correlation of large space structures." 
            Journal of Guidance, Control, and Dynamics 14.2 (1991): 251-259.
        [6] Tcherniak, Dmitri. "Optimal Sensor Placement: a sensor density approach." (2022).
        [7] Papadimitriou, Costas. "Optimal sensor placement methodology for parametric identification of structural systems." 
            Journal of sound and vibration 278.4-5 (2004): 923-947.
        [8] Papadimitriou, Costas, James L. Beck, and Siu-Kui Au. "Entropy-based optimal sensor location for structural model updating." 
            Journal of Vibration and Control 6.5 (2000): 781-800.
        [9] Papadimitriou, Costas. "Optimal sensor placement methodology for parametric identification of structural systems." 
            Journal of sound and vibration 278.4-5 (2004): 923-947.
    """
    try:
        # Vectorized construction of L_mat
        L_mat = (observation_space_node[:, None] == state).astype(np.int8).T
        #Evaluate FIM using Cholesky decomposition
        sigma_C = np.linalg.cholesky(L_mat @ correlation_covariance_matrix @ L_mat.T)
        sigma_A = solve_triangular(sigma_C, (L_mat @ phi), lower= True)
        sing_A = np.linalg.svd(sigma_A, compute_uv= False)
        return np.prod(np.square(sing_A))
    except np.linalg.LinAlgError:
        logging.error("Error in Fisher information matrix computation.")
        raise

args = parser.parse_args()
env_kwargs = {
'sim_modes': args.sim_modes,
'num_sensors': args.num_sensors,
'render': args.render,
'norm': args.norm,
}

pyansys_env = envs_params(env_kwargs)
data = pyansys_env.get_data() 
pyansys_env.close()
mode_shape = data['phi_data']
spatial_cov = data['correlation_covariance_matrix_data']
spatial_cov_iden = np.eye(spatial_cov[(0,)].shape[0])

observation_space_node = data['observation_space_node']
node_id = {}
reward = {}

for comb in mode_shape.keys():
    sensor_indices, efi_vector = effective_independence(mode_shape[comb], 
                                                        env_kwargs.get('num_sensors'), 
                                                        spatial_cov[comb])
    nodes = observation_space_node[sensor_indices]
    node_id[comb] = nodes
    reward[comb] = _info_metric(nodes, 
                                observation_space_node, 
                                spatial_cov[comb], 
                                mode_shape[comb])


# Ensure the 'result' folder exists
result_folder = os.path.join(os.getcwd(), "results")
os.makedirs(result_folder, exist_ok=True)

# Path to save the CSV file
csv_file_path = os.path.join(result_folder, "Effective_Independence_Result_5-sensors_Modes-12345_Covariance_Matrix.csv")

# Write node_id to the CSV file
with open(csv_file_path, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Mode combination", "Optimal Node Id", "Reward"])  # Write header
    for key, value in node_id.items():
        writer.writerow([key, value, reward[key]])  # Write each key-value pair

print(f"CSV file saved at: {csv_file_path}")