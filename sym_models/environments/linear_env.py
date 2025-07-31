import json
from .base_class import *
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import mlflow
from collections import Counter

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
    
class LinearSymbolicEnv(gym.Env):
    def __init__(self, seed=None):
        super().__init__()

        # Every environment will automatically generate an equation for its own
        eq = smu.linear_eq(1,seed=seed)

        self.eq = eq
        self.original_eq = eq

        MAX_NODES = 7
        MAX_EDGES = 6

        self.observation_space = spaces.Dict(
            {
                "nodes": spaces.Box(low=0, high=85, shape=(MAX_NODES, 3), dtype=np.float32),
                "edge_links": spaces.Box(low=0, high=MAX_NODES-1, shape=(MAX_EDGES, 2), dtype=np.int64),
            },
            seed=seed
        )

        # actions = {side, node, action} where side={LHS,RHS}, node={1,2}, action={Transfer,Done}.
        self.action_space = spaces.MultiDiscrete([2, 4, 2], seed=seed)

        self.total_steps = 0
        self.truncating_step = 100

    def _get_obs(self, eq):
        tree = smu.traverse_expr(eq, returnTreeForPlot=True)
        edge_index = np.array([list(tree[i][:2]) for i in range(len(tree)) if i!=0])
        nodes = np.array([node_feature(a[2]) for a in tree])

        # Pad nodes and edges to the max size defined in the observation space.
        padded_nodes = np.zeros(self.observation_space["nodes"].shape, dtype=np.float32)
        padded_nodes[:len(nodes)] = nodes

        padded_edge_links = np.zeros(self.observation_space["edge_links"].shape, dtype=np.int64)
        if len(edge_index) > 0:
            padded_edge_links[:len(edge_index)] = edge_index

        return {
            "nodes": padded_nodes,
            "edge_links": padded_edge_links,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Every time an environment will be reset it will come up with totally new equation
        self.eq = smu.linear_eq(1,seed=seed)  

        # just to check whether our training is going good or not, let's just play with one equation only, and let the model mug up how to solve that
        # self.eq = self.original_eq

        return self._get_obs(self.eq), {"eq": Main.string(self.eq)}

    def step(self, action):
        TRUNCATION_REWARD = -1.5
        COMPLITION_REWARD = 10
        ABSURD_ACTION_REWARD = -0.1
        NOT_COMPLETION_REWARD = -1  # said that it's completed but it wasn't
        STEP_REWARD = -0.03
        terminated, truncated = False, False   # handle the truncated part later on

        # Firstly modify the equation, and then see whether model is saying that our equation has been terminated or not
        # punish the model on taking wrong actions


        new_eq = smu.linear_transport(self.eq, int(action[0]+1), int(action[1]+1))  # check if the model takes some absurd action than what should be done
        reward = -0.5 if Main.string(self.eq)==Main.string(new_eq) else 0
        self.eq = new_eq

        if (action[2]==1):  # model is done
            terminated = smu.linear_termination_status(self.eq)
            if (terminated):
                reward = COMPLITION_REWARD
            else:
                reward = NOT_COMPLETION_REWARD

        self.total_steps += 1
        # reward += self.total_steps * STEP_REWARD
        if (self.total_steps > self.truncating_step):
            truncated = True
            reward += TRUNCATION_REWARD

        return self._get_obs(self.eq), reward, terminated, truncated, {"eq": Main.string(self.eq)}

    def render(self):
        print(f"Current Equation: {Main.string(self.eq)}")

    def close(self):
        pass
