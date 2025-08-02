import torch
from snake_game import SnakeGame
from train_snake_ai import SimpleNet, ACTIONS
import os
import argparse

def load_model(path, device):
    checkpoint = torch.load(path, map_location=device)
    
    # Ensure required keys exist
    required_keys = ['input_size', 'hidden_size', 'output_size', 
                     'model_state_dict', 'grid_width', 'grid_height']
    for key in required_keys:
        if key not in checkpoint:
            raise KeyError(f"Checkpoint is missing required key: '{key}'")

    input_size = checkpoint['input_size']
    hidden_size = checkpoint['hidden_size']
    output_size = checkpoint['output_size']
    grid_width = checkpoint['grid_width']
    grid_height = checkpoint['grid_height']
    
    model = SimpleNet(input_size, hidden_size, output_size, grid_width, grid_height)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, grid_width, grid_height

def play_game(model, grid_width, grid_height, device, fps=60):
    while True:  # Loop to restart game after it's over
        game = SnakeGame(grid_width, grid_height, fps)
        state = game.reset()
        done = False

        while not done:
            state_vec = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                action_index = model(state_vec).argmax().item()
            
            action = ACTIONS[action_index]
            state, _, done, _ = game.step(action)

            game.render()
            game.clock.tick(fps)

            # Handle Pygame quit event to break outer loop gracefully
            for event in game.pygame.event.get():
                if event.type == game.pygame.QUIT:
                    print("Quitting game...")
                    game.pygame.quit()
                    return  # Exit the function to stop the program

        print("Game over. Final score:", game.score)
        print("Restarting game...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the trained Snake AI model.")
    parser.add_argument(
        "-m", "--model_path",
        type=str,
        default="snake_model_30x30.pth",
        help="Path to the trained model (.pth file)."
    )
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Current working directory:", os.getcwd())
    print("Using model:", args.model_path)
    
    if not os.path.isfile(args.model_path):
      raise FileNotFoundError(f"Model file '{args.model_path}' not found. Please train the model first.")    
    
    model, grid_width, grid_height = load_model(args.model_path, device)
    play_game(model, grid_width, grid_height, device)