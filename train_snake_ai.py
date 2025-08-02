import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from snake_game_core import BaseSnakeGame
import pygame
from collections import deque
import numpy as np
import os
import argparse

# ---------- Replay buffer ---------
replay_buffer = deque(maxlen=100_000)
BATCH_SIZE = 64

# ---------- Toggles ----------
ENABLE_PLOT = True
ENABLE_PYGAME = True

# ---------- PyGame ----------
CELL_SIZE = 20
GRID_WIDTH, GRID_HEIGHT = 30, 30
PRINT_EVERY = 10
VISUALIZE_EVERY = 50
VISUALIZE_FPS = 100
WINDOW_WIDTH, WINDOW_HEIGHT = GRID_WIDTH * CELL_SIZE, GRID_HEIGHT * CELL_SIZE

# Colors for rendering
colors = {
    0: (0, 0, 0),        # Empty - Black
    1: (0, 255, 0),      # Snake body - Green
    2: (255, 0, 0),      # Food - Red
    9: (0, 100, 0),      # Snake head - Dark Green
}

def render_grid(screen, grid, score, reward):
    screen.fill((0, 0, 0))
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            color = colors.get(grid[y, x], (255, 255, 255))
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, color, rect)
    pygame.display.set_caption(f"Score: {score} | Reward: {reward:.2f}")
    pygame.display.flip()
    
# ---------- Config ----------
INPUT_SIZE = 11
HIDDEN_SIZE = 64
OUTPUT_SIZE = 4
LR = 0.0005
GAMMA = 0.99
EPSILON_START = 0.1
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
N_EPISODES = 1000

# ---------- Device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- Actions ----------
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)
ACTIONS = [UP, DOWN, LEFT, RIGHT]

# ---------- Model ----------
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, grid_width, grid_height):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)


# ---------- Save Model ----------
def save_model(model, input_size, hidden_size, output_size, grid_width, grid_height):
    if hasattr(model, "_orig_mod"):
        model_to_save = model._orig_mod  # get original model from compiled version
    else:
        model_to_save = model

    # Ensure the 'model/' directory exists
    os.makedirs("models", exist_ok=True)
    
    base_name = f"models/snake_model_{grid_width}x{grid_height}"
    model_name = base_name
    counter = 1

    while os.path.exists(f"{model_name}.pth"):
        model_name = f"{base_name}_{counter}"
        counter += 1
        
    # Save both model state dict and metadata
    checkpoint = {
        'input_size': input_size,
        'hidden_size': hidden_size,
        'output_size': output_size,
        'grid_width': grid_width,
        'grid_height': grid_height,
        'model_state_dict': model_to_save.state_dict()
    }

    torch.save(checkpoint, f"{model_name}.pth")
    print(f"Model saved as {model_name}.pth")

def load_model(path, device):
    checkpoint = torch.load(path, map_location=device)
    
    # Ensure required keys exist
    required_keys = ['input_size', 'hidden_size', 'output_size', 'model_state_dict', 'grid_width', 'grid_height']
    for key in required_keys:
        if key not in checkpoint:
            raise KeyError(f"Checkpoint is missing required key: '{key}'")

    # Extract model parameters and grid size
    input_size = checkpoint['input_size']
    hidden_size = checkpoint['hidden_size']
    output_size = checkpoint['output_size']
    grid_width = checkpoint['grid_width']
    grid_height = checkpoint['grid_height']
    
    # Initialize and load model
    model = SimpleNet(input_size, hidden_size, output_size, grid_width, grid_height)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, input_size, hidden_size, output_size, grid_width, grid_height


def moving_average(data, window):
    return [np.mean(data[max(0, i - window + 1):i + 1]) for i in range(len(data))]


def main(pretrained_model_path=None):
    epsilon = EPSILON_START
    reward_history = []
    score_history = []
    input_size = INPUT_SIZE
    hidden_size = HIDDEN_SIZE
    output_size = OUTPUT_SIZE
    grid_width = GRID_WIDTH
    grid_height = GRID_HEIGHT

    if pretrained_model_path is not None and os.path.isfile(pretrained_model_path):
        print(f"Loading pretrained model from {pretrained_model_path}")
        model, input_size, hidden_size, output_size, grid_width, grid_height = load_model(pretrained_model_path, device)

    else:
        print("No pretrained model provided, training from scratch.")
        model = SimpleNet(input_size, hidden_size, output_size, grid_width, grid_height).to(device)

    if torch.__version__ >= '2.0.0' and device.type == "cuda":
        model = torch.compile(model, backend="aot_eager")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # ---------- PLOT SETUP ----------
    if ENABLE_PLOT:
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.set_title("Total Reward per Episode")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Reward")
        ax2.set_title("Score per Episode")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Score")

    # ---------- PYGAME SETUP ----------
    if ENABLE_PYGAME:
        pygame.init()
        screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        clock = pygame.time.Clock()
    
    try:
        for episode in range(1, N_EPISODES + 1):
            game = BaseSnakeGame(grid_width, grid_height)
            state = game.reset()
            done = False
            total_reward = 0

            while not done:
                state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

                if random.random() < epsilon:
                    action_index = random.randint(0, 3)
                else:
                    with torch.no_grad():
                        q_values = model(state_tensor)
                        action_index = q_values.argmax().item()

                action = ACTIONS[action_index]
                next_state, reward, done, _ = game.step(action)
                total_reward += reward

                replay_buffer.append((state, action_index, reward, next_state, done))
                state = next_state

                # --- Pygame Render ---
                if ENABLE_PYGAME and (episode % VISUALIZE_EVERY == 0):
                    grid = game.get_grid()
                    render_grid(screen, grid, game.score, reward)
                    clock.tick(VISUALIZE_FPS)
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            return

                # --- Training ---
                if len(replay_buffer) >= BATCH_SIZE:
                    batch = random.sample(replay_buffer, BATCH_SIZE)
                    states, actions, rewards, next_states, dones = zip(*batch)

                    states_tensor = torch.from_numpy(np.array(states)).float().to(device)
                    actions_tensor = torch.as_tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
                    rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
                    next_states_tensor = torch.from_numpy(np.array(next_states)).float().to(device)
                    dones_tensor = torch.as_tensor(dones, dtype=torch.bool, device=device).unsqueeze(1)

                    q_values = model(states_tensor).gather(1, actions_tensor)
                    with torch.no_grad():
                        max_next_q = model(next_states_tensor).max(1, keepdim=True)[0]
                        target_q = rewards_tensor + GAMMA * max_next_q * (~dones_tensor)

                    loss = criterion(q_values, target_q)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            reward_history.append(total_reward)
            score_history.append(game.score)
            


            # Inside the training loop, update the scatter plots
            if episode % PRINT_EVERY == 0:
                # Calculate average for the last PRINT_EVERY episodes
                avg_reward = sum(reward_history[-PRINT_EVERY:])
                avg_score = sum(score_history[-PRINT_EVERY:])

                print(f"Episode {episode:5d} | Avg Score: {avg_score:.2f} | Avg Reward: {avg_reward:.2f} | Epsilon: {epsilon:.3f}")

                if ENABLE_PLOT:
                    # Clear previous scatter plot
                    ax1.clear()
                    ax2.clear()

                    # Update scatter data
                    ax1.scatter(range(len(reward_history)), reward_history, label='Reward', color='blue')
                    ax2.scatter(range(len(score_history)), score_history, label='Score', color='orange')

                    # Plot historical average lines with dotted lines
                    avg_rewards = moving_average(reward_history, PRINT_EVERY)
                    avg_scores = moving_average(score_history, PRINT_EVERY)
                    ax1.plot(avg_rewards, linestyle='dotted', color='blue', label='Moving Avg Reward')
                    ax2.plot(avg_scores, linestyle='dotted', color='orange', label='Moving Avg Score')

                    # Set titles and labels again
                    ax1.set_title("Total Reward per Episode")
                    ax1.set_xlabel("Episode")
                    ax1.set_ylabel("Reward")
                    ax2.set_title("Score per Episode")
                    ax2.set_xlabel("Episode")
                    ax2.set_ylabel("Score")

                    # Add legends
                    ax1.legend()
                    ax2.legend()

                    plt.pause(0.001)

            epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        if ENABLE_PLOT:
            plt.ioff()
            plt.show()
      
    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")
        save_model(model, input_size, hidden_size, output_size, grid_width, grid_height)
        return

    save_model(model, input_size, hidden_size, output_size, grid_width, grid_height)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Snake DQN model.")
    parser.add_argument(
        "-m", "--model_path",
        type=str,
        default=None,
        help="Path to a pretrained model .pth file to continue training"
    )
    args = parser.parse_args()
    main(pretrained_model_path=args.model_path)