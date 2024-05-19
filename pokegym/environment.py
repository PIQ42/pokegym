from pathlib import Path
from pdb import set_trace as T
import types
import uuid
from gymnasium import Env, spaces
import numpy as np
from skimage.transform import resize
from collections import defaultdict, deque
import io, os
import random
from pyboy.utils import WindowEvent
import matplotlib.pyplot as plt
from pathlib import Path
import mediapy as media
from pokegym.pyboy_binding import (
    ACTIONS,
    make_env,
    open_state_file,
    load_pyboy_state,
    run_action_on_emulator,
)
from . import ram_map, game_map, data, ram_map_leanke
import subprocess
import multiprocessing
import time
import random
from multiprocessing import Manager
from pokegym import data
from pokegym.constants import *
from enum import IntEnum
from multiprocessing import Manager
from .ram_addresses import RamAddress as RAM

STATE_PATH = __file__.rstrip("environment.py") + "current_state/"

class Base:
    # Shared counter among processes
    counter_lock = multiprocessing.Lock()
    counter = multiprocessing.Value('i', 0)
    
    # Initialize a shared integer with a lock for atomic updates
    shared_length = multiprocessing.Value('i', 0)  # 'i' for integer
    lock = multiprocessing.Lock()  # Lock to synchronize access
    
    # Initialize a Manager for shared BytesIO object
    manager = Manager()
    shared_bytes_io_data = manager.list([b''])  # Holds serialized BytesIO data
    
    def __init__(
        self,
        rom_path="pokemon_red.gb",
        state_path=None,
        headless=True,
        save_video=False,
        quiet=False,
        **kwargs,
    ):
        # Increment counter atomically to get unique sequential identifier
        with Base.counter_lock:
            env_id = Base.counter.value
            Base.counter.value += 1
        """Creates a PokemonRed environment"""
        state_path = STATE_PATH + "Bulbasaudr.state"
        self.game, self.screen = make_env(rom_path, headless, quiet, save_video=False, **kwargs)
        self.initial_states = [open_state_file(state_path)]
        self.headless = headless
        self.mem_padding = 2
        self.memory_shape = 80
        self.use_screen_memory = True
        file_path = 'experiments/running_experiment.txt'
        if not os.path.exists(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as file:
                file.write('default_exp_name')
        # Logging initializations
        with open("experiments/running_experiment.txt", "r") as file:
            exp_name = file.read()
        self.exp_path = Path(f'experiments/{str(exp_name)}')
        self.env_id = env_id
        self.s_path = Path(f'{str(self.exp_path)}/sessions/{str(self.env_id)}')
        self.csv_path = Path(f'./csv')
        self.csv_path.mkdir(parents=True, exist_ok=True)
        self.reset_count = 0
        R, C = self.screen.raw_screen_buffer_dims()
        self.obs_size = (R // 2, C // 2) # 72, 80, 3
        self.screen_memory = defaultdict(
            lambda: np.zeros((255, 255, 1), dtype=np.uint8)
        )
        self.obs_size += (4,)

        self.observation_space = spaces.Box(
            low=0, high=255, dtype=np.uint8, shape=self.obs_size
        )
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.time = 0  
        self.reward_scale = 1
    def load_first_state(self):
        return self.initial_states[0]
    
    def reset(self, seed=None, options=None):
        """Resets the game. Seeding is NOT supported"""
        return self.screen.screen_ndarray(), {}

    def get_fixed_window(self, arr, y, x, window_size):
        height, width, _ = arr.shape
        h_w, w_w = window_size[0], window_size[1]
        h_w, w_w = window_size[0] // 2, window_size[1] // 2
        y_min = max(0, y - h_w)
        y_max = min(height, y + h_w + (window_size[0] % 2))
        x_min = max(0, x - w_w)
        x_max = min(width, x + w_w + (window_size[1] % 2))
        window = arr[y_min:y_max, x_min:x_max]
        pad_top = h_w - (y - y_min)
        pad_bottom = h_w + (window_size[0] % 2) - 1 - (y_max - y - 1)
        pad_left = w_w - (x - x_min)
        pad_right = w_w + (window_size[1] % 2) - 1 - (x_max - x - 1)

        return np.pad(
            window,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
        )

    def render(self):
        if self.use_screen_memory:
            r, c, map_n = ram_map.position(self.game)
            # Update tile map
            mmap = self.screen_memory[map_n]
            if 0 <= r <= 254 and 0 <= c <= 254:
                mmap[r, c] = 255

            # Downsamples the screen and retrieves a fixed window from mmap,
            # then concatenates along the 3rd-dimensional axis (image channel)
            return np.concatenate(
                (
                    self.screen.screen_ndarray()[::2, ::2],
                    self.get_fixed_window(mmap, r, c, self.observation_space.shape),
                ),
                axis=2,
            )
        else:
            return self.screen.screen_ndarray()[::2, ::2]

    def step(self, action):
        run_action_on_emulator(self.game, self.screen, ACTIONS[action], self.headless)
        return self.render(), 0, False, False, {}
        
    def close(self):
        self.game.stop(False)

class Environment(Base):
    def __init__(
        self,
        rom_path="pokemon_red.gb",
        state_path=None,
        headless=True,
        quiet=False,
        verbose=False,
        **kwargs,
    ):
        super().__init__(rom_path, state_path, headless, quiet, **kwargs)
        self.counts_map = np.zeros((444, 436))
        self.verbose = verbose
        self.include_conditions = []
        self.current_maps = []
        self.is_dead = False
        self.last_map = -1
        self.log = True
        self.map_check = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 
        self.seen_pokemon = np.zeros(152, dtype=np.uint8)
        self.caught_pokemon = np.zeros(152, dtype=np.uint8)
        self.moves_obtained = np.zeros(0xA5, dtype=np.uint8)
        self.log = False
        self.rewarded_position = (0, 0)
        self.state_loaded_instead_of_resetting_in_game = 0
        self.badge_count = 0 
        self.is_warping = False

    def update_pokedex(self):
        for i in range(0xD30A - 0xD2F7):
            caught_mem = self.game.get_memory_value(i + 0xD2F7)
            seen_mem = self.game.get_memory_value(i + 0xD30A)
            for j in range(8):
                self.caught_pokemon[8*i + j] = 1 if caught_mem & (1 << j) else 0
                self.seen_pokemon[8*i + j] = 1 if seen_mem & (1 << j) else 0   

    def get_game_coords(self):
        return (ram_map.mem_val(self.game, 0xD362), ram_map.mem_val(self.game, 0xD361), ram_map.mem_val(self.game, 0xD35E))
 
    def current_coords(self):
        return self.last_10_coords[0]
    
    def current_map_id(self):
        return self.last_10_map_ids[0, 0]
    
    def get_badges(self):
        badge_count = ram_map.bit_count(ram_map.mem_val(self.game, 0xD356))
        return badge_count

    def get_levels_sum(self):
        poke_levels = [max(ram_map.mem_val(self.game, a) - 2, 0) for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]]
        return max(sum(poke_levels) - 4, 0) # subtract starting pokemon level
    
    def read_event_bits(self):
        return [
            int(bit)
            for i in range(ram_map.EVENT_FLAGS_START, ram_map.EVENT_FLAGS_START + ram_map.EVENTS_FLAGS_LENGTH)
            for bit in f"{ram_map.read_bit(self.game, i):08b}"
        ]
    
    def read_num_poke(self):
        return ram_map.mem_val(self.game, 0xD163)
    
    def update_num_poke(self):
        self.last_num_poke = self.read_num_poke()
    
    def get_max_n_levels_sum(self, n, max_level):
        num_poke = self.read_num_poke()
        poke_level_addresses = [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
        poke_levels = [max(min(ram_map.mem_val(self.game, a), max_level) - 2, 0) for a in poke_level_addresses[:num_poke]]
        return max(sum(sorted(poke_levels)[-n:]) - 4, 0)
      
    def get_levels_reward(self):
        level_sum = self.get_levels_sum()
        self.max_level_rew = max(self.max_level_rew, level_sum)
        return ((self.max_level_rew - self.party_level_post) * 0.5) + (self.party_level_post * 2.0)
        # return self.max_level_rew * 0.5  # 11/11-3 changed: from 0.5 to 1.0
       
    def get_all_events_reward(self):
        # adds up all event flags, exclude museum ticket
        return max(
            sum(
                [
                    ram_map.bit_count(self.game.get_memory_value(i))
                    for i in range(ram_map.EVENT_FLAGS_START, ram_map.EVENT_FLAGS_START + ram_map.EVENTS_FLAGS_LENGTH)
                ]
            )
            - self.base_event_flags
            - int(ram_map.read_bit(self.game, *ram_map.MUSEUM_TICKET_ADDR)),
            0,
        )
        
    def update_max_event_rew(self):
        cur_rew = self.get_all_events_reward()
        self.max_event_rew = max(cur_rew, self.max_event_rew)
        return self.max_event_rew
       
    def get_badges_reward(self):
        num_badges = self.get_badges()

        if num_badges < 9:
            return num_badges * 5
        elif num_badges < 13:  # env19v2 PPO23
            return 40  # + ((num_badges - 8) * 1)
        else:
            return 40 + 10
        # return num_badges * 5  # env18v4


        # multiply by reward scale
        state_scores = {k: v * self.reward_scale for k, v in state_scores.items()}
        
        return state_scores
        
    def reset(self, seed=None, options=None, max_episode_steps=20480, reward_scale=4.0):
        """Resets the game. Seeding is NOT supported"""
        # BET ADDED
        self.rewarded_events_string = '0' * 2552
        if self.reset_count == 0:
            load_pyboy_state(self.game, self.load_first_state())
        if self.use_screen_memory:
            self.screen_memory = defaultdict(
                lambda: np.zeros((255, 255, 1), dtype=np.uint8)
            )
        self.reset_count += 1
        self.time = 0
        self.max_episode_steps = max_episode_steps
        self.reward_scale = reward_scale
        self.last_reward = None
        self.prev_map_n = None
        self.max_events = 0
        self.max_level_sum = 0
        self.max_opponent_level = 0
        self.seen_coords = set()
        self.last_party_size = 1
        self.seen_pokemon = np.zeros(152, dtype=np.uint8)
        self.caught_pokemon = np.zeros(152, dtype=np.uint8)
        self.seen_coords_no_reward = set()
        self.agent_stats = []
        self.base_explore = 0
        self.max_event_rew = 0
        self.max_level_rew = 0
        self.party_level_base = 0
        self.party_level_post = 0
        self.last_num_mon_in_box = 0
        self.step_count = 0
        self.past_rewards = np.zeros(10240, dtype=np.float32)
        self.rewarded_events_string = '0' * 2552
        self._is_box_mon_higher_level = False
        self.total_reward = 0
        self.rewarded_distances = {} 
        return self.render(), {}

    def step(self, action, fast_video=True):
        run_action_on_emulator(self.game, self.screen, ACTIONS[action], self.headless, fast_video=fast_video,)
        self.time += 1
        # Exploration reward
        r, c, map_n = ram_map.position(self.game)
        self.seen_coords.add((r, c, map_n))
        exploration_reward = 0.1 * len(self.seen_coords) 
        # Level reward
        party_size, party_levels = ram_map.party(self.game)
        self.max_level_sum = max(self.max_level_sum, sum(party_levels))
        level_reward = 1 * self.max_level_sum
        # Badge reward
        badges = ram_map.badges(self.game)
        badges_reward = 20 * badges
        # Event rewards
        events = ram_map.events(self.game)
        self.max_events = max(self.max_events, events)
        event_reward = self.max_events * 3     
        self.update_pokedex()
        seen_pokemon_reward = self.reward_scale * sum(self.seen_pokemon)
        caught_pokemon_reward = self.reward_scale * sum(self.caught_pokemon)        
        reward = self.reward_scale * (
            event_reward
            + seen_pokemon_reward
            + caught_pokemon_reward
            + level_reward
            + badges_reward
            + exploration_reward 
        )

        # Subtract previous reward
        if self.last_reward is None:
            reward = 0
            self.last_reward = 0
        else:
            nxt_reward = reward
            reward -= self.last_reward
            self.last_reward = nxt_reward
            
        info = {}
        done = self.time >= self.max_episode_steps      
        if done or self.time % 10000 == 0:   
            levels = [self.game.get_memory_value(a) for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]]       
            info = {
                "pokemon_exploration_map": self.counts_map, # self.explore_map, #  self.counts_map, 
                "stats": {
                    "step": self.time,
                    "coord": np.sum(self.counts_map),  # np.sum(self.seen_global_coords),
                    "party_size": party_size,
                    "highest_pokemon_level": max(party_levels),
                    "total_party_level": sum(party_levels),
                    "pokemon_exploration_map": self.counts_map,
                    "seen_pokemon": np.sum(self.seen_pokemon),
                    "caught_pokemon": np.sum(self.caught_pokemon),
                },
                "reward": {
                    "delta": reward,
                    "event": event_reward,
                    "level": level_reward,
                    "badges": badges_reward,
                    "exploration": exploration_reward,
                    "seen_pokemon_reward": seen_pokemon_reward,
                    "caught_pokemon_reward": caught_pokemon_reward,
                },
            }       
        return self.render(), reward, done, done, info
