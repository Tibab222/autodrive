# Autonomous Car RL in Rust & Bevy

<p align="center">
  <img src="https://img.shields.io/badge/Rust-000000?logo=rust&logoColor=white" alt="Rust" />
  <img src="https://img.shields.io/badge/Bevy-232326?logo=bevy&logoColor=white" alt="Bevy Engine" />
  <img src="https://img.shields.io/badge/Reinforcement%20Learning-8A2BE2" alt="Reinforcement Learning" />
  <img src="https://img.shields.io/github/license/Tibab222/autodrive" alt="License" />
</p>

A lightweight 2D autonomous parking agent (MLP) trained via **Deep Q-Learning (DQN)** built entirely from scratch in **Rust** using the **Bevy Engine (ECS)**.

No Python bindings, PyTorch, or external RL frameworks—everything including neural network forward/backward passes, raycast lidar sensing, replay buffer, and car dynamics is implemented in native Rust.

<p align="center">
  <img src="docs/demo.gif" alt="Autonomous car reinforcement learning demo" width="800">
</p>

### Commands

| Command                                        | Mode      | Model                                        |
|------------------------------------------------|-----------|----------------------------------------------|
| ```cargo run```                                      | Training  | New randomly initialized model               |
| ```cargo run -- --model=path/to/model.bin```         | Inference | Loads the saved model                        |
| ```cargo run -- --model=path/to/model.bin --train``` | Training  | Loads the saved model and continues training |
| ```cargo run -- --model=path/to/model.bin --train --epsilon=0.5``` | Training | Set the epsilon value when training starts |