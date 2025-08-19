import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import FuncFormatter

def smooth(data, smooth=10):
    """
    Smooth data with moving window average.
    """
    y = np.ones(smooth)
    x = np.asarray(data)
    z = np.ones(len(x))
    smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
    sdt_x = pd.Series(data).rolling(smooth, min_periods=1).std().fillna(0)
    return smoothed_x, sdt_x.values

# Data processing
env_list = [
    "Cantilever-Adversarial-Mode1-v0",
    "Cantilever-Adversarial-Mode2-v0",
    "Cantilever-Adversarial-Mode3-v0",
    "Cantilever-Adversarial-Mode4-v0",
    "Cantilever-Adversarial-Mode5-v0",
    "Cantilever-Adversarial-Mode12-v0",
    "Cantilever-Adversarial-Mode23-v0",
    "Cantilever-Adversarial-Mode34-v0",
    "Cantilever-Adversarial-Mode45-v0",
    "Cantilever-Adversarial-Mode123-v0",
    "Cantilever-Adversarial-Mode234-v0",
    "Cantilever-Adversarial-Mode345-v0",
    "Cantilever-Adversarial-Mode1234-v0",
    "Cantilever-Adversarial-Mode2345-v0",
    "Cantilever-Adversarial-Mode12345-v0"
]

def data_reader(folder, env_list=env_list):
    log_path = r"logs\accel"
    file_name = "logs.csv"
    ptf = os.path.join(log_path, folder, file_name)
    data = pd.read_csv(ptf)
    
    keys = ["test_returns:"]
    reward = {}
    for env in env_list:
        reward[env] = data[keys[0] + env].to_numpy()
    
    total_episode = data['total_episodes'].to_numpy()
    return pd.DataFrame(reward), total_episode

paper_label = {
    "Cantilever-Adversarial-Mode2-v0": 'FSEWM-2', 
    "Cantilever-Adversarial-Mode3-v0": 'FSEWM-3', 
    "Cantilever-Adversarial-Mode4-v0": 'FSEWM-4', 
    "Cantilever-Adversarial-Mode23-v0": 'FSEWM-2,3', 
    "Cantilever-Adversarial-Mode34-v0": 'FSEWM-3,4', 
    "Cantilever-Adversarial-Mode45-v0": 'FSEWM-4,5',
    "Cantilever-Adversarial-Mode123-v0": 'FSEWM-1,2,3', 
    "Cantilever-Adversarial-Mode234-v0": 'FSEWM-2,3,4', 
    "Cantilever-Adversarial-Mode345-v0": 'FSEWM-3,4,5', 
    "Cantilever-Adversarial-Mode2345-v0": 'FSEWM-2,3,4,5', 
    "Cantilever-Adversarial-Mode12345-v0": 'FSEWM-1,2,3,4,5'
}

# Apply Seaborn styles
sns.set_theme(context="paper", style="whitegrid", font_scale=1.5)
sns.set_palette("rocket")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.4


# Environment mapping
env_subplot_mapping = {
    (0, 0): "Cantilever-Adversarial-Mode2-v0",
    (0, 1): "Cantilever-Adversarial-Mode3-v0",
    (0, 2): "Cantilever-Adversarial-Mode4-v0",
    (1, 0): "Cantilever-Adversarial-Mode23-v0",
    (1, 1): "Cantilever-Adversarial-Mode34-v0",
    (1, 2): "Cantilever-Adversarial-Mode45-v0",
    (2, 0): "Cantilever-Adversarial-Mode123-v0",
    (2, 1): "Cantilever-Adversarial-Mode234-v0",
    (2, 2): "Cantilever-Adversarial-Mode345-v0",
    (3, 0): "Cantilever-Adversarial-Mode2345-v0",
    (3, 1): "Cantilever-Adversarial-Mode12345-v0"
}

folder_edi_1 = "Evaluation_Results.csv"
reward_edit_1, total_episode = data_reader(folder_edi_1)

# Create long-form DataFrame for Seaborn
plot_data = []
for env in env_subplot_mapping.values():
    if env in reward_edit_1.columns:
        smoothed, std = smooth(reward_edit_1[env])
        env_data = pd.DataFrame({
            'Episode': total_episode,
            'Return': smoothed,
            'Upper': smoothed + std,
            'Lower': smoothed - std,
            'Environment': paper_label[env]
        })
        plot_data.append(env_data)

plot_df = pd.concat(plot_data)

# Custom formatter for x-axis
def format_episodes(x, pos):
    if x >= 1000:
        return f'{x/1000:.0f}K'
    return f'{x:.0f}'

# Create FacetGrid
g = sns.FacetGrid(plot_df, col='Environment', col_wrap=3, 
                  height=5, aspect=1.2, sharex=True, sharey=False)
g.map_dataframe(lambda data, color, **kws: sns.lineplot(
    x='Episode', y='Return', data=data, color='crimson', linewidth=2.5
))
g.map_dataframe(lambda data, color, **kws: plt.fill_between(
    data['Episode'], data['Lower'], data['Upper'], color='crimson', alpha=0.2
))

# Set titles and labels
g.set_titles(col_template="{col_name}", size=20)
g.set_axis_labels("Episodes", "Return")
g.set(xlim=(0, 135600))

# Apply custom formatter to all x-axes
for ax in g.axes.flat:
    ax.xaxis.set_major_formatter(FuncFormatter(format_episodes))
    ax.set_xticks([0, 30000, 60000, 90000, 120000])
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

# Adjust layout
plt.tight_layout()
g.fig.subplots_adjust(top=0.95, hspace=0.3, wspace=0.2)
#plt.savefig('rl_performance_seaborn.png', dpi=300, bbox_inches='tight')
plt.show()

def data_reader(folder):
    log_path = r"logs\accel"
    file_name = "logs.csv"
    ptf = os.path.join(log_path, folder, file_name)
    data = pd.read_csv(ptf)
    plr_track_level = np.nan_to_num(data['plr_track_level'].to_numpy())
    track_level = np.nan_to_num(data['track_level'].to_numpy())
    levels = plr_track_level + track_level
    agent_plr_max_score = data['agent_plr_max_score'].to_numpy()
    mean_agent_return = data['mean_agent_return'].to_numpy()
    total_episodes = data['total_episodes'].to_numpy()
    df = pd.DataFrame({"total_episodes": total_episodes,"levels": levels, "agent_plr_max_score": agent_plr_max_score, "mean_agent_return":mean_agent_return})
    return df

folder_edi_1 = "Evaluation_Results.csv"
df = data_reader(folder_edi_1)

sns.set(font_scale= 1.8)
sns.set_style("white")
#sns.color_palette("husl", 9)
plt.figure(figsize=(10, 8))
sns.scatterplot(df, x = "total_episodes", y = "mean_agent_return", hue = 'levels', size='levels', sizes=(20, 200), color='crimson')
sns.despine()
#sns.regplot(df, x = "total_episodes", y = "mean_agent_return")
b, a = np.polyfit(df['total_episodes'], df["mean_agent_return"], deg=1)
xseq = np.linspace(0, max(df['total_episodes']), num=len(df['total_episodes']))
df.insert(0, 'reg', a + b * xseq)
df.insert(0, 'xseq', xseq)
#sns.lineplot(df, x= "xseq", y = 'reg', style = 3, linewidth = 3, linestyle = "--")
plt.xlabel("Epsiodes")
plt.ylabel("Mean Return")
plt.grid(0)
plt.show()

plt.figure(figsize=(10, 8))
sns.scatterplot(df, x = "total_episodes", y = "agent_plr_max_score", color='crimson')
sns.despine()
plt.xlabel("Epsiodes")
plt.ylabel("PLR Score")
plt.grid(0)
plt.show()
    
paper_label = {"Cantilever-Adversarial-Mode1-v0": 'FSEWM-1', 
               "Cantilever-Adversarial-Mode5-v0": 'FSEWM-5', 
               "Cantilever-Adversarial-Mode12-v0": 'FSEWM-1,2', 
               "Cantilever-Adversarial-Mode1234-v0":'FSEWM-1,2,3,4'}

env_subplot_mapping = {
    (0, 0): "Cantilever-Adversarial-Mode1-v0",
    (0, 1): "Cantilever-Adversarial-Mode5-v0",
    (0, 2): "Cantilever-Adversarial-Mode12-v0",
    (1, 0): "Cantilever-Adversarial-Mode1234-v0"
}

plot_data = []
for env in env_subplot_mapping.values():
    if env in reward_edit_1.columns:
        smoothed, std = smooth(reward_edit_1[env])
        env_data = pd.DataFrame({
            'Episode': total_episode,
            'Return': smoothed,
            'Upper': smoothed + std,
            'Lower': smoothed - std,
            'Environment': paper_label[env]
        })
        plot_data.append(env_data)

plot_df = pd.concat(plot_data)

# Custom formatter for x-axis
def format_episodes(x, pos):
    if x >= 1000:
        return f'{x/1000:.0f}K'
    return f'{x:.0f}'

# Create FacetGrid
g = sns.FacetGrid(plot_df, col='Environment', col_wrap=2, 
                  height=5, aspect=1.2, sharex=True, sharey=False)
g.map_dataframe(lambda data, color, **kws: sns.lineplot(
    x='Episode', y='Return', data=data, color='crimson', linewidth=2.5
))
g.map_dataframe(lambda data, color, **kws: plt.fill_between(
    data['Episode'], data['Lower'], data['Upper'], color='crimson', alpha=0.2
))

# Set titles and labels
g.set_titles(col_template="{col_name}", size=20)
g.set_axis_labels("Episodes", "Return")
g.set(xlim=(0, 135600))

# Apply custom formatter to all x-axes
for ax in g.axes.flat:
    ax.xaxis.set_major_formatter(FuncFormatter(format_episodes))
    ax.set_xticks([0, 30000, 60000, 90000, 120000])
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

# Adjust layout
plt.tight_layout()
g.fig.subplots_adjust(top=0.95, hspace=0.3, wspace=0.2)
#plt.savefig('rl_performance_seaborn.png', dpi=300, bbox_inches='tight')
plt.show()

#Modal Assurance Criterion
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from util import envs_params
from arguments import parser

def MAC(mode_shape_sim: np.ndarray, 
        mode_shape_mea: np.ndarray,
        wn_exp: list,
        wn_ana: list) -> pd.DataFrame:
    """
    Evaluate Modal Assurance Criterion (MAC) between simulated and measured mode shapes.

    Parameters
    ----------
    mode_shape_sim : np.ndarray
        Simulated mode shapes (modes x sensors).
    mode_shape_mea : np.ndarray
        Measured mode shapes (modes x sensors).
    wn_exp : list
        Experimental natural frequencies.
    wn_ana : list
        Analytical natural frequencies.

    Returns
    -------
    pd.DataFrame
        MAC matrix as a DataFrame.
    """
    mode_length, num_sensors = mode_shape_sim.shape
    assert mode_shape_mea.shape == (mode_length, num_sensors), "Shape mismatch between sim and mea mode shapes."
    MAC_matrix = np.zeros((mode_length, mode_length))
    for i, sim in enumerate(mode_shape_sim):
        for j, mea in enumerate(mode_shape_mea):
            numerator = np.abs(np.dot(mea.T, np.conj(sim))) ** 2
            denominator = np.dot(mea.T, np.conj(mea)) * np.dot(sim.T, np.conj(sim))
            MAC_matrix[i, j] = np.real(numerator / denominator) if denominator != 0 else 0.0
    mac_df = pd.DataFrame(MAC_matrix, index=np.round(wn_exp, 2), columns=np.round(wn_ana, 2))
    return mac_df

# Parse arguments and set up environment
args = parser.parse_args()
env_kwargs = {
    'sim_modes': args.sim_modes,
    'num_sensors': args.num_sensors,
    'render': args.render,
    'norm': args.norm,
}
pyansys_env = envs_params(env_kwargs)
# Read CSV data
filepath = r"dcd-main\eval_results"
file_name = "Evaluation_Results.csv"
data = pd.read_csv(os.path.join(filepath, file_name), index_col=0).T

# Extract node IDs
node_id = {col: data[col][:5].to_numpy() for col in data.columns if col.startswith("node_Id")}
node_id_df = pd.DataFrame(node_id).iloc[:, 5:]

paper_label = {
    "Cantilever-Adversarial-Mode2-v0": 'FSEWM-2', 
    "Cantilever-Adversarial-Mode3-v0": 'FSEWM-3', 
    "Cantilever-Adversarial-Mode4-v0": 'FSEWM-4', 
    "Cantilever-Adversarial-Mode23-v0": 'FSEWM-2,3', 
    "Cantilever-Adversarial-Mode34-v0": 'FSEWM-3,4', 
    "Cantilever-Adversarial-Mode45-v0": 'FSEWM-4,5',
    "Cantilever-Adversarial-Mode123-v0": 'FSEWM-1,2,3', 
    "Cantilever-Adversarial-Mode234-v0": 'FSEWM-2,3,4', 
    "Cantilever-Adversarial-Mode345-v0": 'FSEWM-3,4,5', 
    "Cantilever-Adversarial-Mode2345-v0": 'FSEWM-2,3,4,5', 
    "Cantilever-Adversarial-Mode12345-v0": 'FSEWM-1,2,3,4,5'
}
wn_ana = np.array([ 13.84,  86.61, 150.84,242.84, 338.83])
# Plotting
sns.set(font_scale=1.0)
sns.set_style("whitegrid")
fig, axes = plt.subplots(4, 2, figsize=(15, 15),)
axes = axes.flatten()
i = 0
for col in node_id_df.columns:
    if col[8:] in paper_label.keys():
        phi_ana = pyansys_env.extract_mode_shape(np.array(node_id_df[col]))
        freq_txt = col[35:]
    
        freqs = [int(a)-1 for a in freq_txt[:-3]]
    
        mac_mat = MAC(phi_ana[:len(freqs), :], phi_ana[:len(freqs), :], wn_ana[freqs], wn_ana[freqs])
        sns.heatmap(mac_mat, annot=True, fmt=".2f", cmap='YlGnBu', ax=axes[i])
        axes[i].set_xlabel('Natural Frequency [Hz]')
        axes[i].set_ylabel('Natural Frequency [Hz]')
        axes[i].set_title(paper_label[col[8:]])
        i+=1

plt.tight_layout()
plt.show()

# Read CSV data
filepath = r"Evaluation_Results_path"
file_name = "Evaluation_Results_name.csv"
data = pd.read_csv(os.path.join(filepath, file_name), index_col=0).T

# Extract node IDs
node_id = {col: data[col][:5].to_numpy() for col in data.columns if col.startswith("node_Id")}
node_id_df = pd.DataFrame(node_id).iloc[:, 5:]

paper_label = {"Cantilever-Adversarial-Mode1-v0": 'FSEWM-1', 
               "Cantilever-Adversarial-Mode5-v0": 'FSEWM-5', 
               "Cantilever-Adversarial-Mode12-v0": 'FSEWM-1,2', 
               "Cantilever-Adversarial-Mode1234-v0":'FSEWM-1,2,3,4'}
wn_ana = np.array([ 13.84,  86.61, 150.84,242.84, 338.83])
# Plotting
sns.set(font_scale=1.0)
sns.set_style("whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes = axes.flatten()
i = 0
for col in node_id_df.columns:
    if col[8:] in paper_label.keys():
        phi_ana = pyansys_env.extract_mode_shape(np.array(node_id_df[col]))
        freq_txt = col[35:]
    
        freqs = [int(a)-1 for a in freq_txt[:-3]]
    
        mac_mat = MAC(phi_ana[:len(freqs), :], phi_ana[:len(freqs), :], wn_ana[freqs], wn_ana[freqs])
        sns.heatmap(mac_mat, annot=True, fmt=".2f", cmap='YlGnBu', ax=axes[i])
        axes[i].set_xlabel('Natural Frequency [Hz]')
        axes[i].set_ylabel('Natural Frequency [Hz]')
        axes[i].set_title(paper_label[col[8:]])
        i+=1

plt.tight_layout()
plt.show()
#BarChart
#Train
# Filepath to the CSV file
filepath = r"Evaluation_Results_path"
file_name = "Evaluation_Results_name.csv"

# Read the CSV file
data = pd.read_csv(os.path.join(filepath, file_name), index_col=0)

# Extract solved_rate and reward_metric
solved_rate = data.filter(like="solved_rate", axis=0).iloc[:, 0]
reward_metric = data.filter(like="reward_metric", axis=0)

# Extract environment names and remove the first two words
env_names = list(paper_label.keys())
solve_env_name = ["solved_rate:"+env for env in env_names]
reward_metric_env_name = ["reward_metric:"+env for env in env_names]
solved_rate = solved_rate[solve_env_name]
reward_metric = reward_metric.T[reward_metric_env_name]
reward_metric = reward_metric.T

# Calculate mean and standard deviation of reward_metric
reward_means = reward_metric.mean(axis=1)
reward_stds = reward_metric.std(axis=1)

# Define custom colors for each bar
custom_colors = sns.color_palette("husl", len(env_names))  # "husl" generates distinct colors

# Plot the bar chart using Seaborn
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(12, 6))

# Create the bar plot with error bars
bars = sns.barplot(
    x=env_names,
    y=solved_rate.values,
    palette=custom_colors,  # Use the custom color palette
    edgecolor="black",
    ax=ax,
    ci=None  # Disable Seaborn's default confidence intervals
)

# Add error bars manually
ax.errorbar(
    x=np.arange(len(env_names)),
    y=solved_rate.values,
    yerr=reward_stds.values,  # Use standard deviation as error bars
    fmt='none',  # No markers
    ecolor='black',  # Error bar color
    capsize=5,  # Add caps to the error bars
    capthick=1.5,  # Thickness of the caps
    elinewidth=1.5  # Thickness of the error bars
)

# Add mean and standard deviation as text on each bar
for i, bar in enumerate(ax.patches):
    bar_height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar_height + 0.02,  # Adjust text position slightly above the bar
        f"μ={reward_means[i]:.2f}\nσ={reward_stds[i]:.2f}",
        ha="center",
        va="bottom",
        fontsize=10,
        color="black"
    )

# Customize the plot
ax.set_xlabel("Test Environments", fontsize=12)
ax.set_ylabel("Solved Rate (%)", fontsize=12)
ax.set_xticklabels(list(paper_label.values()), rotation=45, ha="right", fontsize=10)
ax.set_ylim(0,150)
# Add gridlines for better readability
ax.grid(axis="y", linestyle="", alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()

# --- Grouped Bar Chart for Solved Rate (edit-1, edit-3, edit-5) ---
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

env_subplot_mapping = {
    (0, 0): "Cantilever-Adversarial-Mode2-v0",
    (0, 1): "Cantilever-Adversarial-Mode3-v0",
    (0, 2): "Cantilever-Adversarial-Mode4-v0",
    (1, 0): "Cantilever-Adversarial-Mode23-v0",
    (1, 1): "Cantilever-Adversarial-Mode34-v0",
    (1, 2): "Cantilever-Adversarial-Mode45-v0",
    (2, 0): "Cantilever-Adversarial-Mode123-v0",
    (2, 1): "Cantilever-Adversarial-Mode234-v0",
    (2, 2): "Cantilever-Adversarial-Mode345-v0",
    (3, 0): "Cantilever-Adversarial-Mode2345-v0",
    (3, 1): "Cantilever-Adversarial-Mode12345-v0"
}
paper_label = {
    "Cantilever-Adversarial-Mode2-v0": 'FSEWM-2', 
    "Cantilever-Adversarial-Mode3-v0": 'FSEWM-3', 
    "Cantilever-Adversarial-Mode4-v0": 'FSEWM-4', 
    "Cantilever-Adversarial-Mode23-v0": 'FSEWM-2,3', 
    "Cantilever-Adversarial-Mode34-v0": 'FSEWM-3,4', 
    "Cantilever-Adversarial-Mode45-v0": 'FSEWM-4,5',
    "Cantilever-Adversarial-Mode123-v0": 'FSEWM-1,2,3', 
    "Cantilever-Adversarial-Mode234-v0": 'FSEWM-2,3,4', 
    "Cantilever-Adversarial-Mode345-v0": 'FSEWM-3,4,5', 
    "Cantilever-Adversarial-Mode2345-v0": 'FSEWM-2,3,4,5', 
    "Cantilever-Adversarial-Mode12345-v0": 'FSEWM-1,2,3,4,5'
}

# Define experiment files
experiments = {
    "edit-1": "Accel-Mode-12345-5-Sensors-Num-edit-1-0.75-split-88-seed-model-agent.csv",
    "edit-3": "Accel-Mode-12345-5-Sensors-Num-edit-3-0.75-split-88-seed-model-agent.csv",
    "edit-5": "Accel-Mode-12345-5-Sensors-Num-edit-5-0.75-split-88-seed-model-agent.csv"
}

filepath = r"Evaluation_Results_path"
env_names = list(paper_label.keys())
solve_env_name = ["solved_rate:"+env for env in env_names]

# Collect solved rates for each experiment
solved_rates = []
exp_labels = []
for exp_name in experiments:
    file_name = experiments[exp_name]
    data = pd.read_csv(os.path.join(filepath, file_name), index_col=0)
    sr = data.filter(like="solved_rate", axis=0).iloc[:, 0]
    sr = sr[solve_env_name]
    solved_rates.append(sr.values)
    exp_labels.append(exp_name)

solved_rates = np.array(solved_rates)  # shape: (3, num_envs)

# Prepare DataFrame for seaborn
df = pd.DataFrame({
    "Environment": np.tile([paper_label[env] for env in env_names], len(exp_labels)),
    "Solved Rate": solved_rates.flatten(),
    "Experiment": np.repeat(exp_labels, len(env_names))
})

# Plot grouped bar chart
sns.set_theme(style="whitegrid", font_scale=1.8)
fig, ax = plt.subplots(figsize=(18, 12))

sns.barplot(
    data=df,
    x="Environment",
    y="Solved Rate",
    hue="Experiment",
    palette=["slateblue", "violet", "lightpink"],
    edgecolor="black",
    ax=ax
)

ax.set_xlabel("Test Environments", fontsize=20)
ax.set_ylabel("Solved Rate (%)", fontsize=20)
ax.set_xticklabels([paper_label[env] for env in env_names], rotation=45, ha="right", fontsize=18)
ax.set_ylim(0, 130)
ax.legend(fontsize=18)
ax.grid(axis="y", linestyle="", alpha=0.7)

plt.tight_layout()


filepath = r"Evaluation_REsults_path"
file_name = "Evaluation_Result.csv"
data = pd.read_csv(os.path.join(filepath, file_name), index_col=0)
dt_t = data.T
reward_m = {}
for cols in dt_t.columns:
    if cols.startswith("reward_metric"):
        reward_m[cols] = dt_t[cols]

reward_m_df = pd.DataFrame(reward_m)
reward_m_df_mean = reward_m_df.mean(0)
reward_m_df_std = reward_m_df.std(0)

#Ablation Study
def data_reader(folder, env_list = env_list):
    log_path = r"logs\accel"
    file_name = "logs.csv"
    ptf= os.path.join(log_path, folder, file_name)
    data = pd.read_csv(ptf)
    keys = ["test_returns:", "solved_rate:"]
    reward = {}
    solved_rate = {}
    for env in env_list:
        reward[env] = data[keys[0] + env].to_numpy()
        solved_rate[env] = data[keys[1] + env].to_numpy()
        total_episode = data['total_episodes'].to_numpy()
        steps = data["steps"].to_numpy()
    reward_data = pd.DataFrame(reward, index= total_episode)
    solved_rate_data = pd.DataFrame(solved_rate, index = total_episode)
    reward_data.dropna()
    solved_rate_data.dropna()
    return reward_data, solved_rate_data, total_episode, steps

# Define your experiment folders
experiments = {
    "edit-1": "Accel-Mode-12345-5-Sensors-Num-edit-1-0.75-split-model-agent.csv",
    "edit-3": "Accel-Mode-12345-5-Sensors-Num-edit-3-0.75-split-model-agent.csv",
    "edit-5": "Accel-Mode-12345-5-Sensors-Num-edit-5-0.75-split-model-agent.csv"
}
# Load data for all experiments
data = {}
for exp_name, folder in experiments.items():
    reward, solved_rate, total_episode, steps = data_reader(folder)
    data[exp_name] = {
        "reward": reward,
        "solved_rate": solved_rate
    }

# Define color palette for experiments
palette = {
    "edit-1": "crimson",
    "edit-3": "royalblue",
    "edit-5": "forestgreen"
}

# Create figure
sns.set(style="white")
fig, axes = plt.subplots(5, 3, figsize=(18, 16), sharex=True)
#fig.suptitle("Performance Comparison Across Experiments", 
#             fontsize=20, y=0.98, fontweight='bold')

# Environment mapping
env_subplot_mapping = {
    (0,0): "Cantilever-Adversarial-Mode1-v0",
    (0,1): "Cantilever-Adversarial-Mode2-v0",
    (0,2): "Cantilever-Adversarial-Mode3-v0",
    (1,0): "Cantilever-Adversarial-Mode4-v0",
    (1,1): "Cantilever-Adversarial-Mode5-v0",
    (1,2): "Cantilever-Adversarial-Mode12-v0",
    (2,0): "Cantilever-Adversarial-Mode23-v0",
    (2,1): "Cantilever-Adversarial-Mode34-v0",
    (2,2): "Cantilever-Adversarial-Mode45-v0",
    (3,0): "Cantilever-Adversarial-Mode123-v0",
    (3,1): "Cantilever-Adversarial-Mode234-v0",
    (3,2): "Cantilever-Adversarial-Mode345-v0",
    (4,0): "Cantilever-Adversarial-Mode1234-v0",
    (4,1): "Cantilever-Adversarial-Mode2345-v0",
    (4,2): "Cantilever-Adversarial-Mode12345-v0"
}
paper_label = {"Cantilever-Adversarial-Mode1-v0": 'FSEWM-1', 
               "Cantilever-Adversarial-Mode2-v0": 'FSEWM-2', 
               "Cantilever-Adversarial-Mode3-v0": 'FSEWM-3', 
               "Cantilever-Adversarial-Mode4-v0": 'FSEWM-4',
               "Cantilever-Adversarial-Mode5-v0": 'FSEWM-5',
               "Cantilever-Adversarial-Mode12-v0":'FSEWM-1,2', 
               "Cantilever-Adversarial-Mode23-v0":'FSEWM-2,3', 
               "Cantilever-Adversarial-Mode34-v0": 'FSEWM-3,4', 
               "Cantilever-Adversarial-Mode45-v0": 'FSEWM-4,5',
               "Cantilever-Adversarial-Mode123-v0": 'FSEWM-1,2,3', 
               "Cantilever-Adversarial-Mode234-v0": 'FSEWM-2,3,4', 
               "Cantilever-Adversarial-Mode345-v0": 'FSEWM-3,4,5', 
               "Cantilever-Adversarial-Mode1234-v0": 'FSEWM-1,2,3,4', 
               "Cantilever-Adversarial-Mode2345-v0": 'FSEWM-2,3,4,5', 
               "Cantilever-Adversarial-Mode12345-v0": 'FSEWM-1,2,3,4,5'}
# Plot data in each subplot
for (row, col), env in env_subplot_mapping.items():
    ax = axes[row, col]
    
    # Configure spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.grid(False)
    
    # Plot each experiment
    for exp_name in experiments:
        # Get data for this experiment and environment
        df = data[exp_name]["reward"]
        
        # Plot with experiment-specific color
        sns.lineplot(
            data=df,
            x=df.index,
            y=env,
            errorbar='sd',
            ax=ax,
            color=palette[exp_name],
            label=exp_name,
            alpha=0.8
        )
    
    # Set titles and labels
    ax.set_title(f"{paper_label[env]}", fontsize=12)
    ax.set_ylabel("Return", fontsize=10)
    ax.set_xlabel("Total Episodes", fontsize=10)
    
    # Add legend to first subplot only (to avoid repetition)
    if (row, col) == (0, 0):
        ax.legend(title="Experiment", frameon=False)

# Hide unused subplot (3,2)
#axes[3, 2].axis('off')

# Format x-axis ticks to show millions
def format_episodes(x, pos):
    return f'{x/1e6:.0f}M' if x >= 1e6 else f'{x:.0f}'

for i in range(4):
    for j in range(3):
        if (i, j) in env_subplot_mapping:
            axes[i, j].xaxis.set_major_formatter(plt.FuncFormatter(format_episodes))

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.subplots_adjust(hspace=0.3, wspace=0.3)
plt.show()



