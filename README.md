
# 🐍 Snake AI (Reinforcement Learning with PyTorch)

This project implements a reinforcement learning agent that learns to play the classic Snake game using **Deep Q-Learning (DQN)**. The environment is rendered using **Pygame**, and the agent is trained using **PyTorch**.

https://github.com/user-attachments/assets/0e02d945-14a2-4bdc-b5ac-d6971d97fddd

---

## 🚀 Overview

The agent learns to play Snake by trial and error using **Deep Q-learning**. It gets:

* 🟢 **+10** for eating food
* 🔴 **−10** for dying
* ⚪ **−0.1** for every move (to encourage efficient paths)
* ⏱️ **−5.0** if it takes too long to reach food (timeout penalty)

The game logic is handled in a headless environment (BaseSnakeGame) without rendering, making training efficient. A Pygame version is used only for visualization during AI or human play.

The state fed to the neural network includes:
- 🚧 Danger detection (straight, left, right)
- ➡️ Current movement direction (one-hot encoded)
- 🍎 Relative position of food (dx, dy)
- 📏 Manhattan distance to food
- 🐍 Snake length normalized by grid size

This concise state helps the agent learn to differentiate between walls, itself, and food, enabling smarter and more strategic movement.

---

## 🧠 Algorithm

We use the **Q-Learning** update rule:

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]
$$

In Deep Q-Network (DQN), this is approximated as:

```python
target = reward + (0 if done else gamma * torch.max(target_model(next_state)))
```

---

## 📁 Project Structure

```
├── snake_game_core.py         # Core game logic without rendering (used for training)
├── train_snake_ai.py          # Trains the DQN agent with optional pretrained model loading
├── play_snake_ai.py           # Lets a trained AI model play the game (with Pygame rendering)
├── play_snake_human.py        # Allows human to play Snake via keyboard
├── models/                    # Saved PyTorch model files (.pth)
├── assets/
│   └── demo.mp4               # Gameplay demonstration
├── plots/                     # Training performance plots (optional)
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
```

---

## 🏋️‍♂️ Training the Agent

To train the agent **from scratch**:

```bash
python train_snake_ai.py
```

To **resume training** from a previously saved model:

```bash
python train_snake_ai.py --model_path models/snake_model_30x30_trained.pth
```

This loads the weights from the given `.pth` file and continues training.

### ⛔ Keyboard Interrupt

If you stop training with `Ctrl+C`, the model will be saved automatically.

---

## 🎮 Let the Model Play

Once you have a trained model, run:

```bash
python play_snake_ai.py --model_path models/snake_model_30x30_trained.pth
```

This opens a Pygame window where the model plays the game.

---

## 🕹️ Play as Human

Want to play Snake yourself?

```bash
python play_snake_human.py
```

Use arrow keys to control the snake.

---

## 📊 Training Visualization

Training stats are plotted in real time using `matplotlib`:
- 📈 Reward per episode  
- 🧮 Score per episode  
- 📉 Moving averages to visualize learning trends  

Plots refresh live during training, and optional static plots can be saved to the plots/ directory if enabled.

---

## 🛠️ Requirements

Install required packages:

```bash
pip install -r requirements.txt
```

Minimum versions:

* Python 3.9+
* PyTorch >= 1.12
* Pygame
* Matplotlib
* NumPy

---

## 💾 Model Saving Format

Trained models are saved as:

```
snake_model_30x30.pth
snake_model_30x30_1.pth
snake_model_30x30_2.pth
...
```

This avoids overwriting previous models.

---

## 📌 TODO / Improvements

* [ ] Add target network for stable DQN
* [ ] Add experience replay prioritization
* [ ] Implement double DQN
* [ ] Add curriculum learning (e.g., smaller grid → bigger grid)

---

## 📢 License

MIT License © 2025 Usama

---

Let me know if you want a `requirements.txt` or badges (like Python version, license, etc.) added.
