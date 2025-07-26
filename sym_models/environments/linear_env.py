from base_class import *
import gymnasium as gym
from gymnasium import spaces
import numpy as np



class LinearSymbolicEnv(gym.Env):
    def __init__(self, seed=None):
        super().__init__()

        # Every environment will automatically generate an equation for its own
        eq = smu.linear_eq(1,seed=seed)

        self.eq = eq

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
        self.truncating_step = 20

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
        return self._get_obs(self.eq), {"eq": Main.string(self.eq)}

    def step(self, action):
        terminated, truncated = False, False   # handle the truncated part later on

        # Firstly modify the equation, and then see whether model is saying that our equation has been terminated or not
        new_eq = smu.linear_transport(self.eq, int(action[0]+1), int(action[1]+1))  # check if the model takes some absurd action than what should be done
        reward = -0.5 if self.eq==new_eq else 0
        self.eq = new_eq

        if (action[2]==1):  # model is done
            terminated = smu.linear_termination_status(self.eq)
            if (terminated):
                reward = 10
            else:
                reward = -1

        self.total_steps += 1
        if (self.total_steps > self.truncating_step):
            truncated = True
            reward -= 5

        return self._get_obs(self.eq), reward, terminated, truncated, {"eq": Main.string(self.eq)}

    def render(self):
        print(f"Current Equation: {Main.string(self.eq)}")

    def close(self):
        pass
