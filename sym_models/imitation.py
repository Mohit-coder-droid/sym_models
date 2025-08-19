from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.callbacks import BaseCallback,CallbackList
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.utils import obs_as_tensor

from gymnasium import spaces
from typing import Any, Optional, TypeVar, Union
import torch
from collections import OrderedDict
import random
import numpy as np

IM = TypeVar("IM")

class ImitationReplayBuffer():
    def __init__(
            self,
            replay_size: int = 2048,
            replay_transfer_ratio: float = 0.3,
            old_data_ratio: float = 0.2  # prob of getting an old data selected
    ):
        # We can't make a fixed size array, so go with list
        self.data = []  # data collected in the latest rollouts
        self.data_replay = []

        self.replay_size = replay_size
        self.replay_transfer_ratio = replay_transfer_ratio
        self.replay_pos = 0
        self.old_data_ratio = old_data_ratio

    def _transfer_replay(self):
        """To transfer data to the data that has to replayed

        This should happen only when all the data in self.data has undergone imitation training once.

        For now let's do this randomly. Well, I don't think there is any way to select the best data being not computationally expensive (as obs here are graphs), except doing it randomly.
        """
        batch_size = int(len(self.data) * self.replay_transfer_ratio)

        if batch_size:
            batch = random.sample(self.data, batch_size)

            for element in batch:
                if len(self.data_replay) < self.replay_size:
                    self.data_replay.append(element)
                else:
                    self.data_replay[self.replay_pos] = element
                    self.replay_pos = (self.replay_pos + 1) % self.replay_size

    def imitation_training_end(self):
        self._transfer_replay()
        self.data = []

    def analyze(self,data):
        n = len(data)
        if n == 0:
            return []
        remove = [False] * n
        active_occurrence = {}
        process_field = 'prev_eq'

        for i, d in enumerate(data):
            key = d[process_field]
            if key in active_occurrence:
                start = active_occurrence[key]
                for j in range(start, i):
                    remove[j] = True
                keys_to_remove = [k for k, idx in active_occurrence.items() if start <= idx < i]
                for k in keys_to_remove:
                    del active_occurrence[k]
                active_occurrence[key] = i
            else:
                active_occurrence[key] = i

        return [d for i, d in enumerate(data) if not remove[i]]

    def add(self, data):
        self.data.append(self.analyze(data))

    def get(self):
        if random.random()<self.old_data_ratio and len(self.data_replay):
            return random.choice(self.data_replay)
        else:
            return random.choice(self.data)
        
class ImitationDataCallback(BaseCallback):
    """
    A custom callback that records observations, rewards, and actions for each episode.
    """
    def __init__(self, imitation_buffer ,verbose=0, termination_steps:int=101):
        super(ImitationDataCallback, self).__init__(verbose)

        # Datasets to store the current rollouts environment (this is not an expert it has to be reset)
        self.current = []  # a series of S(state),A(Action),R(reward) and other info

        self.imitation_buffer = imitation_buffer

    def _on_step(self) -> bool:
        """
        This method is called after each step in the environment.
        """
        self.current.append({
            "next_obs": self.locals['new_obs'],
            "obs": self.locals['infos'][0]['prev_obs'],
            "rewards": self.locals['rewards'],
            "actions": self.locals['actions'],
            "dones": self.locals['dones'],
            "eq": self.locals['infos'][0]['eq'],
            "prev_eq": self.locals['infos'][0]['prev_eq'],
            "locals":self.locals
        })

        # Check if the episode is done
        # why is there a difference between self.locals['dones'] and 'terminated'
        # if self.locals['dones'] and len(self.current)!=self.termination_steps:
        if self.locals['infos'][0]['terminated']:
            self.imitation_buffer.add(self.current)

        if self.locals['dones'] or self.locals['infos'][0]['TimeLimit.truncated']:
            # Reset for the next episode
            self.current= []


        return True

class IM(PPO):
    """Imitating Model"""
    def __init__(
        self,
        policy: Union[str, type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[dict[str, Any]] = None,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
        ):

        super().__init__(
            policy,
            env,
            learning_rate = learning_rate,
            n_steps = n_steps,
            batch_size = batch_size,
            n_epochs = n_epochs,
            gamma = gamma,
            gae_lambda = gae_lambda,
            clip_range = clip_range,
            clip_range_vf= clip_range_vf,
            normalize_advantage = normalize_advantage,
            ent_coef = ent_coef,
            vf_coef = vf_coef,
            max_grad_norm = max_grad_norm,
            use_sde = use_sde,
            sde_sample_freq = sde_sample_freq,
            rollout_buffer_class = rollout_buffer_class,
            rollout_buffer_kwargs = rollout_buffer_kwargs,
            target_kl = target_kl,
            stats_window_size = stats_window_size,
            tensorboard_log = tensorboard_log,
            policy_kwargs = policy_kwargs,
            verbose = verbose,
            seed = seed,
            device = device,
            _init_setup_model = _init_setup_model,
        )

        self.imitation_buffer = ImitationReplayBuffer()
        self.imitation_callback = ImitationDataCallback(self.imitation_buffer,verbose=0)

    def _make_orderedDict(self, obs_dict):
        return OrderedDict([(k,np.expand_dims(obs_dict[k], axis=0)) for k in list(obs_dict.keys())])

    def collect_rollouts_imitation(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:

        # Make some callback functions to give some values to the callback

        # make a better sampling
        n_steps = 0
        rollout_buffer.reset()

        callback.on_rollout_start()

        data = self.imitation_buffer.get()

        while n_steps < n_rollout_steps:
            for env_steps in data:
                obs = self._make_orderedDict(env_steps['obs'])
                with torch.no_grad():
                    obs_tensor = obs_as_tensor(obs, self.device)
                    actions, values, log_probs = self.policy(obs_tensor)
                actions = actions.cpu().numpy()

                self.num_timesteps += env.num_envs
                n_steps += 1
                if n_steps>n_rollout_steps:
                    break

                if np.array_equal(actions, env_steps['actions']):
                    rewards = np.array([1], dtype=np.float32)
                else:
                    rewards = np.array([0], dtype=np.float32)

                # what are they doing of this info buffer
                infos = env_steps['locals']['infos']
                dones = env_steps['dones']

                self._update_info_buffer(infos, dones)

                if isinstance(self.action_space, spaces.Discrete):
                    # Reshape in case of discrete action
                    actions = actions.reshape(-1, 1)


                # what they are doing inside this
                for idx, done in enumerate(dones):
                    if (
                        done
                        and infos[idx].get("terminal_observation") is not None
                        and infos[idx].get("TimeLimit.truncated", False)
                    ):
                        terminal_obs = self.policy.obs_to_tensor(self._make_orderedDict(infos[idx]["terminal_observation"]))[0]
                        with torch.no_grad():
                            terminal_value = self.policy.predict_values(terminal_obs)[0]  
                        rewards[idx] += self.gamma * terminal_value

                self.rollout_buffer.add(
                        obs,  
                        actions,
                        rewards,
                        self._last_episode_starts,  # whether this is a start of new episode, what does the model has to do with it if the observations are MDP. It's nowhere is used in training process in case of PPO
                        values,
                        log_probs,
                    )

                self._last_episode_starts = dones   # check whether this is correct or not

            data = self.imitation_buffer.get()

        # Check for any discrepancy while calculating advantage
        with torch.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(data[-1]['next_obs'], self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    def learn(
        self: IM,
        total_timesteps: int,
        callback = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
        imitation_ratio : float = 0.2
    ) -> IM:
        iteration = 0

        # Add by default the imitation callback
        callback = CallbackList([callback, self.imitation_callback])

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        # gives all the local and global variables to the callback function
        callback.on_training_start(locals(), globals())

        assert self.env is not None

        # self.num_timesteps is getting updated in self.collect_rollouts in every step that the model is performing 
        while self.num_timesteps < total_timesteps:

            # Both the types of training can't be done simultaneously, one has to be done more than other. So how much preference one has to be given with respect to other.
            # There comes the imitation ratio
            imitation_training = False
            if random.random() < imitation_ratio and len(self.imitation_buffer.data):   # imitate the model
                imitation_training = True
                self.logger.record("train/train_type", "Imitating")
                continue_training = self.collect_rollouts_imitation(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

            else:   # make the model explore
                self.logger.record("train/train_type", "Exploring")
                continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

            if not continue_training:
                break

            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            iteration += 1
            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                self.dump_logs(iteration)

            self.train()

            if imitation_training:
                self.imitation_buffer.imitation_training_end()

        callback.on_training_end()

        return self
