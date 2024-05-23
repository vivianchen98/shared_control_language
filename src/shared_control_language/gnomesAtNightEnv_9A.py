import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame


class GnomesAtNightEnv9A(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, round=1):
        self.round = round
        self.n_agents = 2
        self.current_player = 0
        self.size = 9  # The size of the mazes

        # Define the rendering variables
        self.CELL_SIZE = 40
        self.WALL_WIDTH = 3
        self.OUT_WALL_WIDTH = 5
        self.OFFSET = 20  # Space between the two mazes
        self.MARGIN = 20  # Margin around the mazes
        self.SCREEN_WIDTH = (
            (self.size * self.CELL_SIZE) * 2 + self.OFFSET + (self.MARGIN * 2)
        )
        self.SCREEN_HEIGHT = self.size * self.CELL_SIZE + (self.MARGIN * 2)
        self.WALL_COLOR = (0, 0, 0)  # black
        self.BG_COLOR = (255, 255, 255)  # white

        # Define the maze layouts
        self.size = 9  # The size of the square grid
        walls = [
            # zero_walls
            [
                # vertical
                {"from": {"x": 2, "y": 1}, "to": {"x": 3, "y": 1}},
                {"from": {"x": 2, "y": 2}, "to": {"x": 3, "y": 2}},
                {"from": {"x": 2, "y": 4}, "to": {"x": 3, "y": 4}},
                {"from": {"x": 2, "y": 5}, "to": {"x": 3, "y": 5}},
                {"from": {"x": 2, "y": 8}, "to": {"x": 3, "y": 8}},
                {"from": {"x": 3, "y": 6}, "to": {"x": 4, "y": 6}},
                {"from": {"x": 3, "y": 7}, "to": {"x": 4, "y": 7}},
                {"from": {"x": 4, "y": 3}, "to": {"x": 5, "y": 3}},
                {"from": {"x": 4, "y": 4}, "to": {"x": 5, "y": 4}},
                {"from": {"x": 4, "y": 5}, "to": {"x": 5, "y": 5}},
                {"from": {"x": 4, "y": 8}, "to": {"x": 5, "y": 8}},
                {"from": {"x": 4, "y": 9}, "to": {"x": 5, "y": 9}},
                {"from": {"x": 5, "y": 1}, "to": {"x": 6, "y": 1}},
                {"from": {"x": 5, "y": 3}, "to": {"x": 6, "y": 3}},
                {"from": {"x": 5, "y": 6}, "to": {"x": 6, "y": 6}},
                {"from": {"x": 5, "y": 7}, "to": {"x": 6, "y": 7}},
                {"from": {"x": 6, "y": 5}, "to": {"x": 7, "y": 5}},
                {"from": {"x": 6, "y": 8}, "to": {"x": 7, "y": 8}},
                {"from": {"x": 7, "y": 1}, "to": {"x": 8, "y": 1}},
                # {'from': {'x': 7, 'y': 2}, 'to': {'x': 8, 'y': 2}},
                {"from": {"x": 7, "y": 3}, "to": {"x": 8, "y": 3}},
                {"from": {"x": 7, "y": 4}, "to": {"x": 8, "y": 4}},
                {"from": {"x": 7, "y": 5}, "to": {"x": 8, "y": 5}},
                {"from": {"x": 7, "y": 7}, "to": {"x": 8, "y": 7}},
                # horizontal
                {"from": {"x": 3, "y": 2}, "to": {"x": 3, "y": 3}},
                {"from": {"x": 4, "y": 2}, "to": {"x": 4, "y": 3}},
                # {'from': {'x': 5, 'y': 2}, 'to': {'x': 5, 'y': 3}},
                {"from": {"x": 6, "y": 2}, "to": {"x": 6, "y": 3}},
                {"from": {"x": 7, "y": 2}, "to": {"x": 7, "y": 3}},
                {"from": {"x": 9, "y": 2}, "to": {"x": 9, "y": 3}},
                {"from": {"x": 1, "y": 3}, "to": {"x": 1, "y": 4}},
                {"from": {"x": 2, "y": 3}, "to": {"x": 2, "y": 4}},
                {"from": {"x": 6, "y": 4}, "to": {"x": 6, "y": 5}},
                {"from": {"x": 8, "y": 4}, "to": {"x": 8, "y": 5}},
                {"from": {"x": 9, "y": 4}, "to": {"x": 9, "y": 5}},
                {"from": {"x": 2, "y": 5}, "to": {"x": 2, "y": 6}},
                {"from": {"x": 3, "y": 5}, "to": {"x": 3, "y": 6}},
                {"from": {"x": 4, "y": 5}, "to": {"x": 4, "y": 6}},
                {"from": {"x": 6, "y": 5}, "to": {"x": 6, "y": 6}},
                {"from": {"x": 7, "y": 5}, "to": {"x": 7, "y": 6}},
                {"from": {"x": 1, "y": 7}, "to": {"x": 1, "y": 8}},
                {"from": {"x": 2, "y": 7}, "to": {"x": 2, "y": 8}},
                {"from": {"x": 3, "y": 7}, "to": {"x": 3, "y": 8}},
                {"from": {"x": 4, "y": 7}, "to": {"x": 4, "y": 8}},
                {"from": {"x": 5, "y": 7}, "to": {"x": 5, "y": 8}},
                {"from": {"x": 6, "y": 7}, "to": {"x": 6, "y": 8}},
                {"from": {"x": 7, "y": 7}, "to": {"x": 7, "y": 8}},
                {"from": {"x": 8, "y": 7}, "to": {"x": 8, "y": 8}},
                {"from": {"x": 9, "y": 7}, "to": {"x": 9, "y": 8}},
            ],
            # one_walls
            [
                # vertical
                {"from": {"x": 2, "y": 2}, "to": {"x": 3, "y": 2}},
                {"from": {"x": 2, "y": 3}, "to": {"x": 3, "y": 3}},
                {"from": {"x": 2, "y": 5}, "to": {"x": 3, "y": 5}},
                {"from": {"x": 2, "y": 6}, "to": {"x": 3, "y": 6}},
                {"from": {"x": 2, "y": 7}, "to": {"x": 3, "y": 7}},
                {"from": {"x": 3, "y": 5}, "to": {"x": 4, "y": 5}},
                {"from": {"x": 3, "y": 8}, "to": {"x": 4, "y": 8}},
                {"from": {"x": 3, "y": 9}, "to": {"x": 4, "y": 9}},
                {"from": {"x": 4, "y": 1}, "to": {"x": 5, "y": 1}},
                {"from": {"x": 4, "y": 2}, "to": {"x": 5, "y": 2}},
                {"from": {"x": 4, "y": 3}, "to": {"x": 5, "y": 3}},
                {"from": {"x": 4, "y": 4}, "to": {"x": 5, "y": 4}},
                {"from": {"x": 4, "y": 6}, "to": {"x": 5, "y": 6}},
                {"from": {"x": 5, "y": 3}, "to": {"x": 6, "y": 3}},
                {"from": {"x": 5, "y": 5}, "to": {"x": 6, "y": 5}},
                {"from": {"x": 5, "y": 9}, "to": {"x": 6, "y": 9}},
                {"from": {"x": 6, "y": 6}, "to": {"x": 7, "y": 6}},
                {"from": {"x": 6, "y": 7}, "to": {"x": 7, "y": 7}},
                {"from": {"x": 7, "y": 2}, "to": {"x": 8, "y": 2}},
                {"from": {"x": 7, "y": 3}, "to": {"x": 8, "y": 3}},
                {"from": {"x": 7, "y": 4}, "to": {"x": 8, "y": 4}},
                {"from": {"x": 7, "y": 8}, "to": {"x": 8, "y": 8}},
                {"from": {"x": 7, "y": 9}, "to": {"x": 8, "y": 9}},
                # horizontal
                {"from": {"x": 1, "y": 2}, "to": {"x": 1, "y": 3}},
                {"from": {"x": 2, "y": 2}, "to": {"x": 2, "y": 3}},
                {"from": {"x": 3, "y": 2}, "to": {"x": 3, "y": 3}},
                {"from": {"x": 4, "y": 2}, "to": {"x": 4, "y": 3}},
                {"from": {"x": 5, "y": 2}, "to": {"x": 5, "y": 3}},
                {"from": {"x": 6, "y": 2}, "to": {"x": 6, "y": 3}},
                {"from": {"x": 7, "y": 2}, "to": {"x": 7, "y": 3}},
                {"from": {"x": 8, "y": 3}, "to": {"x": 8, "y": 4}},
                {"from": {"x": 9, "y": 3}, "to": {"x": 9, "y": 4}},
                {"from": {"x": 1, "y": 4}, "to": {"x": 1, "y": 5}},
                {"from": {"x": 2, "y": 4}, "to": {"x": 2, "y": 5}},
                {"from": {"x": 4, "y": 4}, "to": {"x": 4, "y": 5}},
                {"from": {"x": 3, "y": 5}, "to": {"x": 3, "y": 6}},
                {"from": {"x": 4, "y": 5}, "to": {"x": 4, "y": 6}},
                {"from": {"x": 5, "y": 5}, "to": {"x": 5, "y": 6}},
                {"from": {"x": 6, "y": 5}, "to": {"x": 6, "y": 6}},
                {"from": {"x": 7, "y": 5}, "to": {"x": 7, "y": 6}},
                {"from": {"x": 8, "y": 5}, "to": {"x": 8, "y": 6}},
                {"from": {"x": 9, "y": 5}, "to": {"x": 9, "y": 6}},
                {"from": {"x": 2, "y": 7}, "to": {"x": 2, "y": 8}},
                {"from": {"x": 3, "y": 7}, "to": {"x": 3, "y": 8}},
                {"from": {"x": 4, "y": 7}, "to": {"x": 4, "y": 8}},
                {"from": {"x": 5, "y": 7}, "to": {"x": 5, "y": 8}},
                {"from": {"x": 6, "y": 7}, "to": {"x": 6, "y": 8}},
                {"from": {"x": 7, "y": 7}, "to": {"x": 7, "y": 8}},
                {"from": {"x": 8, "y": 7}, "to": {"x": 8, "y": 8}},
            ],
        ]
        self.one_hot_walls = self.encode_walls_as_one_hot(walls)

        # Define position of token and treasure
        self.token_pos = np.array((0, 0), dtype=int)
        treasures = [(4, 0), (6, 2), (3, 3), (4, 5), (3, 7)]
        self.treasure_pos_this_round = treasures[self.round - 1]
        self.treasure = {
            "onWhichSide": (self.round - 1) % 2,
            "pos": np.array(self.treasure_pos_this_round),
        }

        # Define the observation space: (149, 2) array
        # 149 = |current_player| + |token_pos| + |walls|, |treasure_pos| = 1 + 2 + 8*9*2 + 2 for each player
        obs_low = np.array([0] + [0, 0] + [0] * 144 + [-1, -1], dtype=np.int32)
        obs_high = np.array([1] + [8, 8] + [1] * 144 + [8, 8], dtype=np.int32)
        self.observation_space = spaces.Tuple(
            [
                gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.int32)
                for _ in range(self.n_agents)
            ]
        )

        # Define the action space for { 0: stay_put, 1: right, 2: up, 3: left, 4: down }
        self.action_space = spaces.Discrete(5)

        # Define the mapping from actions to directions
        self._action_to_direction = {
            0: np.array([0, 0]),  # stay_put
            1: np.array([1, 0]),  # right
            2: np.array([0, -1]),  # up
            3: np.array([-1, 0]),  # left
            4: np.array([0, 1]),  # down
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    """ Game dynamics: turn-based, two-player, cooperative control """

    def step(self, action):
        # Update the token position based on the action if valid
        if self.is_action_valid(self.current_player, action):
            self.token_pos = self.token_pos + self._action_to_direction[action]
        # print('player ', self.current_player, 'token_pos: ', self.token_pos)

        # get updated observations, rewards, terminations, and infos
        terminated = np.array_equal(self.token_pos, self.treasure["pos"])
        reward = self._get_rewards(action)
        obs_n = self.get_observations()
        info = self.get_infos()
        truncated = False

        # switch turn between players 0 and 1 if not done
        if not terminated and not truncated:
            self.current_player = 1 - self.current_player

        if self.render_mode == "human":
            self._render_frame()

        # Return the new obs, reward, done, and any additional info
        return obs_n, reward, terminated, truncated, info

    def reset(self, seed=None):
        # Reset the token position to (1, 1)
        self.current_player = 0
        self.token_pos = np.array((0, 0), dtype=int)
        self.treasure = {
            "onWhichSide": (self.round - 1) % 2,
            "pos": np.array(self.treasure_pos_this_round),
        }

        # Set the random seed if one is provided
        if seed is not None:
            np.random.seed(seed)

        # randomize the token and treasure positions
        self.current_player = np.random.randint(0, 2)
        self.treasure["pos"] = np.random.randint(0, 9, size=2, dtype=int)
        while np.array_equal(self.token_pos, self.treasure["pos"]):
            self.token_pos = np.random.randint(0, 9, size=2, dtype=int)

        # reset the random seed to None to avoid affecting other parts of the program
        np.random.seed(None)

        # render the initial frame if in human mode
        if self.render_mode == "human":
            self._render_frame()

        obs_n = self.get_observations()
        info = self.get_infos()
        return obs_n, info

    def is_final(self):
        return np.array_equal(self.token_pos, self.treasure["pos"])

    """ Helper functions """

    def is_action_inbound(self, action):
        # Assuming self.size is the size of the grid
        max_index = self.size - 1

        # Moving right from the rightmost column
        if action == 1 and self.token_pos[0] == max_index:
            return False

        # Moving up from the topmost row
        if action == 2 and self.token_pos[1] == 0:
            return False

        # Moving left from the leftmost column
        if action == 3 and self.token_pos[0] == 0:
            return False

        # Moving down from the bottom row
        if action == 4 and self.token_pos[1] == max_index:
            return False

        return True

    def is_action_valid(self, current_player, action):
        # Assuming self.size is the size of the grid
        max_index = self.size - 1

        # Stay put is always valid
        if action == 0:
            return True

        # Moving right
        if action == 1:
            if self.token_pos[0] == max_index:
                # print("invalid: out of bounds; Retry!")
                return False
            # Check for a vertical wall to the right of the token
            elif (
                self.one_hot_walls[current_player]["vertical"][
                    self.token_pos[1], self.token_pos[0]
                ]
                == 1
            ):
                # print("wall on the right")
                return False

        # Moving up
        if action == 2:
            if self.token_pos[1] == 0:
                # print("invalid: out of bounds; Retry!")
                return False
            # Check for a horizontal wall above the token
            if (
                self.one_hot_walls[current_player]["horizontal"][
                    self.token_pos[1] - 1, self.token_pos[0]
                ]
                == 1
            ):
                # print("wall above")
                return False

        # Moving left
        if action == 3:
            if self.token_pos[0] == 0:
                # print("invalid: out of bounds; Retry!")
                return False
            # Check for a vertical wall to the left of the token
            if (
                self.one_hot_walls[current_player]["vertical"][
                    self.token_pos[1], self.token_pos[0] - 1
                ]
                == 1
            ):
                # print("wall on the left")
                return False

        # Moving down
        if action == 4:
            if self.token_pos[1] == max_index:
                # print("invalid: out of bounds; Retry!")
                return False
            # Check for a horizontal wall below the token
            if (
                self.one_hot_walls[current_player]["horizontal"][
                    self.token_pos[1], self.token_pos[0]
                ]
                == 1
            ):
                # print("wall below")
                return False

        # If none of the above conditions are met, the action is valid
        return True

    def get_valid_actions(self, current_player):
        return [a for a in range(5) if self.is_action_valid(current_player, a)]

    def get_inbound_actions(self):
        return [a for a in range(5) if self.is_action_inbound(a)]

    def _get_rewards(self, action):
        reward = 0

        # if reach the treasure, reward = 10 (to encourage reaching the treasure)
        if np.array_equal(self.token_pos, self.treasure["pos"]):
            reward += 20

        # compute the shortest distance from the token to the treasure

        # Scaled reward based on Manhattan distance to treasure (wrong! because of the walls)
        # reward -= 0.1 * self.get_infos()["distance"]

        # if hit a wall, reward = -0.5 (to discourage hitting walls)
        if not self.is_action_valid(self.current_player, action):
            reward -= 0.5

        # each step, reward = -0.1 (to encourage shorter paths)
        reward -= 0.1

        return reward

    def get_observations(self):
        _obs = []
        for player in range(self.n_agents):
            _obs.append(self.get_player_observation(player))
        return tuple(_obs)

    def get_player_observation(self, player):
        # Set treasure position to (-1, -1) if not visible; else, set to actual position
        if self.treasure["onWhichSide"] == player:
            treasure_pos = np.array(self.treasure["pos"])
        else:
            treasure_pos = np.array((-1, -1))

        # flatten the one_hot_walls matrices
        walls = self.one_hot_walls[player]
        flattened_walls = np.concatenate(
            (walls["vertical"].flatten(), walls["horizontal"].flatten())
        )

        _player_i_obs = [self.current_player]  # current_player
        _player_i_obs += self.token_pos.tolist()  # token_pos
        _player_i_obs += flattened_walls.tolist()  # walls
        _player_i_obs += treasure_pos.tolist()  # treasure_pos

        return np.array(_player_i_obs, dtype=np.int32)

    def get_infos(self):
        return {
            "current_player": self.current_player,
            "token": self.token_pos,
            "treasure": self.treasure,
            "distance": np.linalg.norm(
                self.token_pos - self.treasure["pos"],
                ord=1,  # Manhattan distance from token to treasure
            ),
        }

    def encode_walls_as_one_hot(self, walls):
        one_hot_walls = []

        for wall_group in walls:
            # Initialize the numpy arrays for vertical and horizontal walls
            vertical_walls = np.zeros((9, 8), dtype=int)
            horizontal_walls = np.zeros((8, 9), dtype=int)

            for wall in wall_group:
                # Calculate the wall indices, adjusting for 0-indexing
                wall_index_x = wall["from"]["x"] - 1
                wall_index_y = wall["from"]["y"] - 1

                # Check if the wall is vertical
                if wall["from"]["y"] == wall["to"]["y"]:
                    # Wall is vertical
                    vertical_walls[wall_index_y, wall_index_x] = 1
                else:
                    # Wall is horizontal
                    horizontal_walls[wall_index_y, wall_index_x] = 1

            # Append the wall matrices for the current player to the one_hot_walls list
            one_hot_walls.append(
                {"vertical": vertical_walls, "horizontal": horizontal_walls}
            )

        return one_hot_walls

    def decode_player_wall_obs(self, player_obs_walls):
        # player_obs_walls is a flattened array of shape (2 * 9 * 8,)
        # The first 9 * 8 elements correspond to the vertical walls
        # The next 9 * 8 elements correspond to the horizontal walls
        # The following code reshapes the array into two matrices of shape (9, 8)
        # The first matrix corresponds to the vertical walls
        #
        # The second matrix corresponds to the horizontal walls
        vertical_walls = player_obs_walls[: 9 * 8].reshape((9, 8))
        horizontal_walls = player_obs_walls[9 * 8 :].reshape((8, 9))
        one_hot_walls = {"vertical": vertical_walls, "horizontal": horizontal_walls}
        return one_hot_walls

    """ Rendering """

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        canvas.fill(self.BG_COLOR)

        self.draw_walls_offset(
            canvas,
            self.one_hot_walls[0],
            offset_x=0,
            MARGIN=self.MARGIN,
            CELL_SIZE=self.CELL_SIZE,
            OUT_WALL_WIDTH=self.OUT_WALL_WIDTH,
            WALL_WIDTH=self.WALL_WIDTH,
            WALL_COLOR=self.WALL_COLOR,
        )
        self.draw_walls_offset(
            canvas,
            self.one_hot_walls[1],
            offset_x=self.size * self.CELL_SIZE + self.OFFSET,
            MARGIN=self.MARGIN,
            CELL_SIZE=self.CELL_SIZE,
            OUT_WALL_WIDTH=self.OUT_WALL_WIDTH,
            WALL_WIDTH=self.WALL_WIDTH,
            WALL_COLOR=self.WALL_COLOR,
        )

        if self.current_player == 0:
            solid_center_x = int(
                (self.token_pos[0] + 0.5) * self.CELL_SIZE + self.MARGIN
            )
            empty_center_x = int(
                (self.token_pos[0] + 0.5) * self.CELL_SIZE
                + self.MARGIN
                + self.size * self.CELL_SIZE
                + self.OFFSET
            )
        else:
            solid_center_x = int(
                (self.token_pos[0] + 0.5) * self.CELL_SIZE
                + self.MARGIN
                + self.size * self.CELL_SIZE
                + self.OFFSET
            )
            empty_center_x = int(
                (self.token_pos[0] + 0.5) * self.CELL_SIZE + self.MARGIN
            )

        # Draw the token in solid line for the current player
        pygame.draw.circle(
            canvas,
            (255, 0, 0),
            (
                solid_center_x,
                int((self.token_pos[1] + 0.5) * self.CELL_SIZE + self.MARGIN),
            ),
            int(self.CELL_SIZE / 3),
        )

        # draw token in dashed line for the other player
        pygame.draw.circle(
            canvas,
            (255, 0, 0),
            (
                empty_center_x,
                int((self.token_pos[1] + 0.5) * self.CELL_SIZE + self.MARGIN),
            ),
            int(self.CELL_SIZE / 3),
            3,
        )

        # Draw the treasure
        if self.treasure["onWhichSide"] == 0:
            pygame.draw.circle(
                canvas,
                (255, 255, 0),
                (
                    int((self.treasure["pos"][0] + 0.5) * self.CELL_SIZE + self.MARGIN),
                    int((self.treasure["pos"][1] + 0.5) * self.CELL_SIZE + self.MARGIN),
                ),
                int(self.CELL_SIZE / 3),
            )
        elif self.treasure["onWhichSide"] == 1:
            pygame.draw.circle(
                canvas,
                (255, 255, 0),
                (
                    int(
                        (self.treasure["pos"][0] + 0.5) * self.CELL_SIZE
                        + self.MARGIN
                        + self.size * self.CELL_SIZE
                        + self.OFFSET
                    ),
                    int((self.treasure["pos"][1] + 0.5) * self.CELL_SIZE + self.MARGIN),
                ),
                int(self.CELL_SIZE / 3),
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def draw_walls_offset(
        self,
        screen,
        one_hot_walls,
        offset_x,
        MARGIN,
        CELL_SIZE,
        OUT_WALL_WIDTH,
        WALL_WIDTH,
        WALL_COLOR,
    ):
        offset_x += MARGIN
        offset_y = MARGIN  # Define the vertical offset as well

        # Draw the outer boundary of the maze
        pygame.draw.rect(
            screen,
            WALL_COLOR,
            (offset_x, offset_y, self.size * CELL_SIZE, OUT_WALL_WIDTH),
        )  # Top
        pygame.draw.rect(
            screen,
            WALL_COLOR,
            (offset_x, offset_y, OUT_WALL_WIDTH, self.size * CELL_SIZE),
        )  # Left
        pygame.draw.rect(
            screen,
            WALL_COLOR,
            (
                offset_x,
                offset_y + self.size * CELL_SIZE - OUT_WALL_WIDTH,
                self.size * CELL_SIZE,
                OUT_WALL_WIDTH,
            ),
        )  # Bottom
        pygame.draw.rect(
            screen,
            WALL_COLOR,
            (
                offset_x + self.size * CELL_SIZE - OUT_WALL_WIDTH,
                offset_y,
                OUT_WALL_WIDTH,
                self.size * CELL_SIZE,
            ),
        )  # Right

        # Draw the internal walls of the maze without exceeding the outer boundary
        for y in range(self.size):
            for x in range(self.size - 1):
                if one_hot_walls["vertical"][y][x] == 1:
                    # Draw vertical walls within the cell boundaries
                    pygame.draw.line(
                        screen,
                        WALL_COLOR,
                        ((x + 1) * CELL_SIZE + offset_x, y * CELL_SIZE + offset_y),
                        (
                            (x + 1) * CELL_SIZE + offset_x,
                            (y + 1) * CELL_SIZE + offset_y,
                        ),
                        WALL_WIDTH,
                    )

        for y in range(self.size - 1):
            for x in range(self.size):
                if one_hot_walls["horizontal"][y][x] == 1:
                    # Draw horizontal walls within the cell boundaries
                    pygame.draw.line(
                        screen,
                        WALL_COLOR,
                        (x * CELL_SIZE + offset_x, (y + 1) * CELL_SIZE + offset_y),
                        (
                            (x + 1) * CELL_SIZE + offset_x,
                            (y + 1) * CELL_SIZE + offset_y,
                        ),
                        WALL_WIDTH,
                    )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
