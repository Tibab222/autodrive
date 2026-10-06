mod state;
mod model;
mod train;
pub mod agent_car;

use crate::agent::{model::NeuralNetwork, train::Trainer};

use bevy::ecs::resource::Resource;
pub use state::{CarAction, encode_state};
pub use train::{ReplayBuffer, Transition};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentMode {
    Training,
    Inference
}

#[derive(Resource)]
pub struct Agent {
    pub(crate) mode: AgentMode,
    pub trainer: Option<Trainer>,
    pub model: Option<NeuralNetwork>,
    pub epoch: usize,
    pub last_loss: f32,
    pub loss_history: Vec<f32>,
    pub episode_reward: f32,
    pub episode_steps: usize,
    pub episode_distance_sum: f32,
    pub total_episodes: usize,
    pub successful_episodes: usize,
    pub crashed_episodes: usize,
    pub best_success_rate: f32,
    pub average_episode_reward: f32,
    pub average_episode_steps: f32,
    pub average_distance: f32,
    pub last_max_q: f32,
}

impl Agent {
    // ***************** INIT FUNCTIONS *****************
    pub fn new(mode: AgentMode) -> Self {
        let model = NeuralNetwork::new_random();

        match mode {
            AgentMode::Training => {
                let trainer = Trainer::new(10000, model.clone());
                Agent {
                    mode, trainer: Some(trainer), model: Some(model), epoch: 0, last_loss: 0.0,
                    loss_history: Vec::new(), episode_reward: 0.0, episode_steps: 0,
                    episode_distance_sum: 0.0, total_episodes: 0, successful_episodes: 0,
                    crashed_episodes: 0, best_success_rate: 0.0, average_episode_reward: 0.0, average_episode_steps: 0.0,
                    average_distance: 0.0, last_max_q: 0.0,
                }
            },
            AgentMode::Inference => {
                Agent {
                    mode, trainer: None, model: Some(model), epoch: 0, last_loss: 0.0,
                    loss_history: Vec::new(), episode_reward: 0.0, episode_steps: 0,
                    episode_distance_sum: 0.0, total_episodes: 0, successful_episodes: 0,
                    crashed_episodes: 0, best_success_rate: 0.0, average_episode_reward: 0.0, average_episode_steps: 0.0,
                    average_distance: 0.0, last_max_q: 0.0,
                }
            },
        }
    }

    pub fn load_model(path: &str, mode: AgentMode) -> std::io::Result<Self> {
        let model = model::NeuralNetwork::load_from_file(path)?;
        if mode == AgentMode::Training {
            let trainer = Trainer::new(10000, model.clone());
            Ok(Agent { mode, trainer: Some(trainer), model: Some(model), epoch: 0, last_loss: 0.0, loss_history: Vec::new(), episode_reward: 0.0, episode_steps: 0, episode_distance_sum: 0.0, total_episodes: 0, successful_episodes: 0, crashed_episodes: 0, best_success_rate: 0.0, average_episode_reward: 0.0, average_episode_steps: 0.0, average_distance: 0.0, last_max_q: 0.0 })
        } else {
            Ok(Agent { mode, trainer: None, model: Some(model), epoch: 0, last_loss: 0.0, loss_history: Vec::new(), episode_reward: 0.0, episode_steps: 0, episode_distance_sum: 0.0, total_episodes: 0, successful_episodes: 0, crashed_episodes: 0, best_success_rate: 0.0, average_episode_reward: 0.0, average_episode_steps: 0.0, average_distance: 0.0, last_max_q: 0.0 })
        }
    }

    pub fn save_model(&self, path: &str) {
        if let Some(trainer) = &self.trainer {
            trainer.network.save_to_file(path).expect("Failed to save model to file");
        } else if let Some(model) = &self.model {
            model.save_to_file(path).expect("Failed to save model to file");
        } else {
            panic!("No model available to save");
        }
    }

    // ***************** INFERENCE & TRAINING FUNCTIONS *****************

    // function to perform inference using the model or trainer (no exploration)
    pub fn infer(&self, x: &[f32]) -> Vec<f32> {
        let q_values = match self.mode {
            AgentMode::Training => {
                if let Some(trainer) = &self.trainer {
                    trainer.network.forward(x)
                } else {
                    panic!("Trainer is not initialized in training mode");
                }
            },
            AgentMode::Inference => {
                if let Some(model) = &self.model {
                    model.forward(x)
                } else {
                    panic!("Model is not initialized in inference mode");
                }
            },
        };
        q_values.iter().map(|&q| q as f32).collect()
    }

    /// selects an action based on the current state (epsilon-greedy for training & Greedy for inference).
    pub fn select_action(&mut self, state: &[f32]) -> usize {
        match self.mode {
            AgentMode::Training => {
                if let Some(trainer) = &mut self.trainer {
                    trainer.select_action(state)
                } else {
                    panic!("Trainer is not initialized in training mode");
                }
            },
            AgentMode::Inference => {
                if let Some(model) = &self.model {
                    let q_values = model.forward(state);
                    let mut max_q = f32::NEG_INFINITY;
                    let mut best_action = 0;

                    for (i, &q) in q_values.iter().enumerate() {
                        if q > max_q {
                            max_q = q;
                            best_action = i;
                        }
                    }
                    best_action
                } else {
                    panic!("Model is not initialized in inference mode");
                }
            },
        }
    }

    pub fn step_train(&mut self) -> f32 {
        if let Some(trainer) = &mut self.trainer {
            let loss = trainer.train_step();
            self.last_loss = loss;
            if loss > 0.0 {
                self.loss_history.push(loss);
                if self.loss_history.len() > 120 {
                    self.loss_history.remove(0);
                }
            }
            loss
        } else {
            0.0
        }
    }
}