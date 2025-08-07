import json
from .base_class import *
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import mlflow
from collections import Counter
import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, ModuleList
from torch_geometric.nn import TransformerConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.data import Data, Batch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class ActionStateCallback(BaseCallback):
    """
    A custom callback to log state, action, and next_state.

    :param log_freq: Log every `log_freq` calls to `_on_step()`.
    :param log_dir: Path to the directory where to save the data.
    :param verbose: Verbosity level.
    """
    def __init__(self, log_freq: int, log_dir: str = "/Data/sandeep/zMohit/RL Models/Equation Model/logs", verbose: int = 0):
        super(ActionStateCallback, self).__init__(verbose)
        self.log_freq = log_freq
        self.log_dir = log_dir
        self.data = []  # List to store the (state, action, next_state) tuples
        # The observation before the action was taken
        self.last_obs = None
    
    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        We capture the initial observation here.
        """
        # Get the initial observation from the environment
        # self.training_env.reset() returns a tuple of (obs, info)
        self.last_obs = self.training_env.reset()

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        
        :return: (bool) If the callback returns False, training is aborted.
        """
        # Check if it's time to log
        if self.n_calls % self.log_freq == 0:
            # `self.locals` contains many useful variables from the training loop
            # `actions` is the action taken by the policy
            action = self.locals["actions"]
            # `new_obs` is the observation returned by the environment after the step
            new_obs = self.locals["infos"]

            # Store the data
            transition = {
                "state": self.last_obs,
                "action": action,
                "next_state": new_obs
            }
            self.data.append(transition)

            if self.verbose > 0:
                print(f"Step {self.n_calls}: Logged a transition.")

        # Update last_obs to the current observation for the next step
        self.last_obs = self.locals["infos"].copy()
        
        return True # Continue training

    def _on_training_end(self) -> None:
        """
        This method is called at the end of training.
        We'll save our collected data here.
        """
        os.makedirs(self.log_dir, exist_ok=True)
        save_path = os.path.join(self.log_dir, "action_state_log.npy")
        print(f"Saving collected data to {save_path}")
        np.save(save_path, self.data)

class EpisodeEndCallback(BaseCallback):
    """
    A custom callback to log the total reward and reason for episode end.
    
    :param verbose: Verbosity level.
    """
    def __init__(self,log_dir, verbose: int = 0):
        super(EpisodeEndCallback, self).__init__(verbose)
        self.episode_data = [] # List to store dicts of {"reward": val, "reason": str}
        self.log_dir = log_dir
    
    def _on_step(self) -> bool:
        """
        This method is called after each step in the environment.
        It checks for episode ends and logs the required info.
        """
        # self.locals['dones'] is a boolean array, True for envs that just finished
        for i, done in enumerate(self.locals["dones"]):
            if done:
                # Episode has just finished for the i-th environment
                info = self.locals["infos"][i]
                
                # The 'episode' key is added by the VecEnv wrappers when an episode ends
                # It contains the total reward 'r', length 'l', and time 't'
                total_reward = info["episode"]["r"]
                
                # Determine if the episode was terminated or truncated
                # "TimeLimit.truncated" is True if truncated by a time limit
                if info.get("TimeLimit.truncated", False):
                    reason = "Truncated"
                else:
                    # Otherwise, it was terminated (e.g., pole fell in CartPole)
                    reason = "Terminated"
                
                # Log the data
                self.episode_data.append({"reward": total_reward, "reason": reason})
                
                if self.verbose > 0:
                    print(f"Episode ended. Reward: {total_reward}, Reason: {reason}")
        
        return True # Continue training
    
    def _on_training_end(self) -> None:
        """
        This method is called at the end of training.
        We'll save our collected data here.
        """
        os.makedirs(self.log_dir, exist_ok=True)
        save_path = os.path.join(self.log_dir, "termination_reward_log.json")
        print(f"Saving collected data to {save_path}")
        json.dump(self.episode_data,open(save_path, "w"))

        # Aggregate logging
        rewards = [ep["reward"] for ep in self.episode_data]
        reasons = [ep["reason"] for ep in self.episode_data]
        reason_counts = Counter(reasons)

        # Log to MLflow
        if mlflow.active_run() is not None:
            mlflow.log_metric("episode_count", len(self.episode_data))
            mlflow.log_metric("average_reward", np.mean(rewards))
            mlflow.log_metric("max_reward", np.max(rewards))
            mlflow.log_metric("min_reward", np.min(rewards))
            mlflow.log_metric("terminated_count", reason_counts.get("Terminated", 0))
            mlflow.log_metric("truncated_count", reason_counts.get("Truncated", 0))

            # Optional: log the full JSON as artifact
            # mlflow.log_artifact(save_path)

        else:
            print("No active MLflow run found - skipping MLflow logging.")

class GNN(BaseFeaturesExtractor):
    def __init__(self,observation_space, model_params, features_dim: int = 256):
        super().__init__(observation_space,features_dim)
        embedding_size = model_params["model_embedding_size"]
        n_heads = model_params["model_attention_heads"]
        self.n_layers = model_params["model_layers"]
        dropout_rate = model_params["model_dropout_rate"]
        # top_k_ratio = model_params["model_top_k_ratio"]
        self.top_k_every_n = model_params["model_top_k_every_n"]
        # dense_neurons = model_params["model_dense_neurons"]
        edge_dim = model_params["model_edge_dim"]
        self.global_node = model_params["global_node"]   # whether to have a global node or not

        feature_size = observation_space['nodes'].shape[1]

        self.conv_layers = ModuleList([])
        self.transf_layers = ModuleList([])
        self.bn_layers = ModuleList([])

        # Transformer layers (Convolving by making multi head attention, which has to be concatenated together, so size self.conv1 = (feature_size, n_heads*embedding_size))
        self.conv1 = TransformerConv(feature_size, embedding_size, heads=n_heads, dropout=dropout_rate, edge_dim=edge_dim, beta=True)

        self.transf1 = Linear(embedding_size*n_heads, embedding_size)
        self.bn1 = BatchNorm1d(embedding_size)

        # Other layers
        for i in range(self.n_layers):
            self.conv_layers.append(TransformerConv(embedding_size,
                                                    embedding_size,
                                                    heads=n_heads,
                                                    dropout=dropout_rate,
                                                    edge_dim=edge_dim,
                                                    beta=True))

            self.transf_layers.append(Linear(embedding_size*n_heads, embedding_size))
            self.bn_layers.append(BatchNorm1d(embedding_size))  # do batch normalization

            # Linear layers
        self.linear1 = Linear(2*embedding_size, features_dim)

    def _forward(self, x, edge_index, edge_attr, batch_index) -> torch.tensor:
        # Initial transformation
        x = self.conv1(x, edge_index)
        x = torch.relu(self.transf1(x))
        x = self.bn1(x)

        # Holds the intermediate graph representations
        global_representation = []

        for i in range(self.n_layers):
            x = self.conv_layers[i](x, edge_index)
            x = torch.relu(self.transf_layers[i](x))
            x = self.bn_layers[i](x)
            
            if self.global_node:
                global_representation.append(torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1))

        if self.global_node:
            x = sum(global_representation)

        # Output block
        x = torch.relu(self.linear1(x))
        return x

    def forward(self, observation):
        # Do batching of the data in pytorch_geometric's way
        data_list = []

        for i in range(observation["nodes"].size(0)):  # For each graph in the batch
            x = observation["nodes"][i]                # (7, 3)
            edge_attr = observation['edge_attributes'][i]   # (6, 1)
            edge_index = observation['edge_links'][i].to(torch.int64).T  # (2, 6), transpose from (6, 2)

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            data_list.append(data)

        batch = Batch.from_data_list(data_list)

        return self._forward(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    
class LinearSymbolicEnv(gym.Env):
    def __init__(self,reward_params, seed=None):
        super().__init__()
        self.reward_params  =reward_params

        # Every environment will automatically generate an equation for its own
        # Now let's train it over all the linear environments 
        self.type = 5
        eq = smu.linear_eq(self.type,seed=seed)

        self.eq = eq
        self.original_eq = eq

        MAX_NODES = 25    # 7 for simple linear equation
        MAX_EDGES = 24

        self.observation_space = spaces.Dict(
            {
                "nodes": spaces.Box(low=0, high=85, shape=(MAX_NODES, 3), dtype=np.float32),
                "edge_links": spaces.Box(low=0, high=1, shape=(MAX_EDGES, 2), dtype=np.int64),
                "edge_attributes":spaces.Box(low=0,high=0, shape=(MAX_EDGES,1),dtype=np.int32)
            },
            seed=seed
        )

        # actions = {side, node, action} where side={LHS,RHS}, node={1,2}, action={Transfer,Done}.
        self.action_space = spaces.MultiDiscrete([2, 4, 2], seed=seed)

        self.total_steps = 0
        self.truncating_step = 100

    def _get_obs(self, eq):
        tree = smu.traverse_expr(eq, returnTreeForPlot=True)
        edge_index = np.array([list(tree[i][:2]) for i in range(len(tree)) if i!=0], dtype=np.int64)
        nodes = np.array([node_feature(a[2]) for a in tree], dtype=np.float32)

        # No need for padding I think
        # Pad nodes and edges to the max size defined in the observation space.
        padded_nodes = np.zeros(self.observation_space["nodes"].shape, dtype=np.float32)
        padded_nodes[:len(nodes)] = nodes

        padded_edge_links = np.zeros(self.observation_space["edge_links"].shape, dtype=np.int64)
        if len(edge_index) > 0:
            padded_edge_links[:len(edge_index)] = edge_index

        return {
            "nodes": padded_nodes,
            "edge_links": padded_edge_links,
            "edge_attributes": np.zeros((len(padded_edge_links), 1))
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Every time an environment will be reset it will come up with totally new equation
        self.eq = smu.linear_eq(self.type,seed=seed)
        self.total_steps = 0

        # just to check whether our training is going good or not, let's just play with one equation only, and let the model mug up how to solve that
        # self.eq = self.original_eq

        return self._get_obs(self.eq), {"eq": Main.string(self.eq)}

    def step(self, action):
        TRUNCATION_REWARD = self.reward_params['TRUNCATION_REWARD']
        COMPLETION_REWARD = self.reward_params['COMPLETION_REWARD']
        ABSURD_ACTION_REWARD = self.reward_params['ABSURD_ACTION_REWARD']
        NOT_COMPLETION_REWARD = self.reward_params['NOT_COMPLETION_REWARD']  # said that it's completed but it wasn't
        STEP_REWARD = self.reward_params['STEP_REWARD']
        terminated, truncated = False, False   # handle the truncated part later on

        # Firstly modify the equation, and then see whether model is saying that our equation has been terminated or not
        # punish the model on taking wrong actions
        reward = 0










        # Check why this error is coming
#         RuntimeError: <PyCall.jlwrap (in a Julia function called from Python)
# JULIA: MethodError: no method matching /(::Nothing, ::SymbolicUtils.BasicSymbolic{Real})
# The function `/` exists, but no method is defined for this combination of argument types.

# Closest candidates are:
#   /(!Matched::MutableArithmetics.Zero, ::Any)
#    @ MutableArithmetics ~/.julia/packages/MutableArithmetics/tNSBd/src/rewrite.jl:74
#   /(::Any, !Matched::ChainRulesCore.NotImplemented)
#    @ ChainRulesCore ~/.julia/packages/ChainRulesCore/XAgYn/src/tangent_types/notimplemented.jl:43
#   /(!Matched::PyObject, ::Any)
#    @ PyCall ~/.julia/packages/PyCall/1gn3u/src/pyoperators.jl:13
#   ...

# Stacktrace:
#   [1] linear_transport(expr_tree::Vector{Tuple{Function, AbstractVector{Any}}}, side::Int64, node::Int64)
#     @ SymbolicModelsUtils ~/zMohit/RL Models/SymbolicModelsUtils.jl/src/Equation model/linear_model.jl:27
#   [2] linear_transport(expr::Symbolics.Equation, side::Int64, node::Int64)
#     @ SymbolicModelsUtils ~/zMohit/RL Models/SymbolicModelsUtils.jl/src/Equation model/linear_model.jl:64
#   [3] invokelatest(::Any, ::Any, ::Vararg{Any}; kwargs::@Kwargs{})
#     @ Base ./essentials.jl:1055

        try:
            new_eq = smu.linear_transport(self.eq, int(action[0]+1), int(action[1]+1))
        except:
            new_eq = self.eq
        
        # if the size of nodes get increased from the max size, that means the model has done stupid to reach at this point, don't change the equation after this
        tree = smu.traverse_expr(new_eq, returnTreeForPlot=True)
        if len(tree)>self.observation_space["nodes"].shape[0]:
            new_eq = self.eq
        
        
         # check if the model takes some absurd action than what should be done
        reward = ABSURD_ACTION_REWARD if Main.string(self.eq)==Main.string(new_eq) else 0


        self.eq = new_eq

        if (action[2]==1):  # model is done
            terminated = smu.linear_termination_status(self.eq)
            if (terminated):
                reward = COMPLETION_REWARD
            else:
                reward = NOT_COMPLETION_REWARD


        # This reward is the main reason for the slow training
        self.total_steps += 1
        reward += self.total_steps * STEP_REWARD
        
        if (self.total_steps > self.truncating_step):
            truncated = True
            reward += TRUNCATION_REWARD

        return self._get_obs(self.eq), reward, terminated, truncated, {"eq": Main.string(self.eq)}

    def render(self):
        print(f"Current Equation: {Main.string(self.eq)}")

    def close(self):
        pass
