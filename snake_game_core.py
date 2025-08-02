import random
from collections import deque
import numpy as np

# Directions
UP, DOWN, LEFT, RIGHT = (0, -1), (0, 1), (-1, 0), (1, 0)
DIRECTIONS = [UP, DOWN, LEFT, RIGHT]

# Directional turns
RIGHT_TURNS = {UP: RIGHT, RIGHT: DOWN, DOWN: LEFT, LEFT: UP}
LEFT_TURNS = {UP: LEFT, LEFT: DOWN, DOWN: RIGHT, RIGHT: UP}

class BaseSnakeGame:
    __slots__ = ("snake", "snake_set", "direction", "food", "score", 
                 "game_over", "total_steps", "GRID_WIDTH", "GRID_HEIGHT",
                 "steps_since_last_food")

    def __init__(self, GRID_WIDTH = 30, GRID_HEIGHT = 40):
      self.GRID_WIDTH = GRID_WIDTH
      self.GRID_HEIGHT = GRID_HEIGHT 
      self.reset()

    def reset(self):
        mid = (self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)
        self.snake = deque([mid])
        self.snake_set = {mid}
        self.direction = RIGHT
        self.food = self._place_food()
        self.score = 0
        self.game_over = False
        self.total_steps = 0
        self.steps_since_last_food = 0
        return self._get_state()

    def _place_food(self):
        while True:
            food = (random.randint(0, self.GRID_WIDTH - 1), random.randint(0, self.GRID_HEIGHT - 1))
            if food not in self.snake_set:
                return food

    def _check_collision(self, pos):
        x, y = pos
        return (
            x < 0 or x >= self.GRID_WIDTH or
            y < 0 or y >= self.GRID_HEIGHT or
            pos in self.snake_set
        )
        
    def _get_state(self):
        head_x, head_y = self.snake[0]
        dir_x, dir_y = self.direction

        right = RIGHT_TURNS[self.direction]
        left = LEFT_TURNS[self.direction]

        def danger(pos):
            return float(self._check_collision(pos))

        # Danger in 3 directions
        danger_straight = danger((head_x + dir_x, head_y + dir_y))
        danger_right    = danger((head_x + right[0], head_y + right[1]))
        danger_left     = danger((head_x + left[0], head_y + left[1]))

        # Direction one-hot
        direction_state = [
            float(self.direction == UP),
            float(self.direction == DOWN),
            float(self.direction == LEFT),
            float(self.direction == RIGHT)
        ]

        # Food relative position (normalized)
        food_dx = (self.food[0] - head_x) / self.GRID_WIDTH
        food_dy = (self.food[1] - head_y) / self.GRID_HEIGHT

        # Manhattan distance to food (normalized by max distance)
        manhattan_dist = (abs(self.food[0] - head_x) + abs(self.food[1] - head_y)) / (self.GRID_WIDTH + self.GRID_HEIGHT)

        # Snake length (normalized by max possible length)
        max_snake_length = self.GRID_WIDTH * self.GRID_HEIGHT
        snake_length = len(self.snake) / max_snake_length

        state_array = np.array([
            danger_straight, danger_right, danger_left,
            *direction_state,
            food_dx, food_dy,
            manhattan_dist,
            snake_length
        ], dtype=np.float32)

        return state_array

    def get_grid(self):
        """Returns a 2D NumPy array representing the game state (for visualization)."""
        grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=np.uint8)

        for x, y in self.snake:
            grid[y, x] = 1  # Snake body
        head_x, head_y = self.snake[0]
        grid[head_y, head_x] = 9  # Snake head
        food_x, food_y = self.food
        grid[food_y, food_x] = 2  # Food

        return grid

    def step(self, action):
        if (action[0] * -1, action[1] * -1) != self.direction:
            self.direction = action

        head_x, head_y = self.snake[0]
        dir_x, dir_y = self.direction
        new_head = (head_x + dir_x, head_y + dir_y)

        self.game_over = self._check_collision(new_head)
        if self.game_over:
            return self._get_state(), -10.0, True, {"score": self.score}

        self.snake.appendleft(new_head)
        self.snake_set.add(new_head)

        if new_head == self.food:
            self.score += 1
            reward = 10.0
            self.food = self._place_food()
            self.steps_since_last_food = 0
        else:
            tail = self.snake.pop()
            self.snake_set.remove(tail)
            reward = -0.1
            self.steps_since_last_food += 1
            
            if self.steps_since_last_food > (self.GRID_WIDTH + self.GRID_HEIGHT):
              self.game_over = True
              reward = -5.0
              return self._get_state(), reward, True, {"score": self.score}

        return self._get_state(), reward, False, {"score": self.score}

    def render(self):
        raise NotImplementedError("Rendering is only available in the GUI version.")

    def run_human_play(self):
        raise NotImplementedError("Human play is only supported with rendering.")