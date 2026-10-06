use rand::{RngExt, seq::IndexedRandom};

use crate::agent::model;

/// Q-learning method for training the neural network

const EPSILON_DECAY: f32 = 0.995; // Decay rate for epsilon
const MIN_EPSILON: f32 = 0.05; // Minimum value for epsilon
const BATCH_SIZE: usize = 64;
const GAMMA: f32 = 0.99; // Discount factor for future rewards
const LEARNING_RATE: f32 = 0.0005; // Learning rate for the optimizer
const MAX_TARGET_Q: f32 = 100.0;
const TARGET_UPDATE_INTERVAL: usize = 100; // Update target network every 1000 training steps

#[derive(Clone)]
pub struct Transition {
    pub state: Vec<f32>,
    pub action: usize,
    pub reward: f32,
    pub next_state: Vec<f32>,
    pub done: bool,
}

pub struct ReplayBuffer {
    buffer: Vec<Transition>,
    capacity: usize,
    position: usize,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            capacity,
            position: 0,
        }
    }

    pub fn push(&mut self, transition: Transition) {
        if self.buffer.len() < self.capacity {
            self.buffer.push(transition);
        } else {
            self.buffer[self.position] = transition;
            self.position = (self.position + 1) % self.capacity;
        }
    }

    pub fn sample(&self, batch_size: usize) -> Vec<Transition> {
        let mut rng = rand::rng();
        self.buffer
            .sample(&mut rng, batch_size)
            .cloned()
            .collect()
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }
}

pub struct Trainer {
    pub network: model::NeuralNetwork,
    target_network: model::NeuralNetwork,
    pub replay_buffer: ReplayBuffer,
    pub epsilon: f32, // Exploration rate
    pub epsilon_decay: f32,
    pub min_epsilon: f32,
    training_steps: usize,
}

impl Trainer {
    pub fn new(buffer_capacity: usize, model: model::NeuralNetwork) -> Self {
        let target_network = model.clone();
        Self {
            network: model,
            target_network: target_network,
            replay_buffer: ReplayBuffer::new(buffer_capacity),
            epsilon: 1.0,
            epsilon_decay: EPSILON_DECAY,
            min_epsilon: MIN_EPSILON,
            training_steps: 0,
        }
    }

    pub fn set_epsilon(&mut self, epsilon: f32) {
        self.epsilon = epsilon;
    }

    /// Selects an action based on the current state using an epsilon-greedy policy.
    pub fn select_action(&mut self, state: &[f32]) -> usize {
        let mut rng = rand::rng();

        // Epsilon-greedy action selection
        if rng.random::<f32>() < self.epsilon {
            let idx = rng.random_range(0..model::OUTPUT_SIZE);
            return idx;
        }

        let q_values = self.network.forward(state);
        let mut max_q = f32::NEG_INFINITY;
        let mut best_action = 0;

        for (i, &q) in q_values.iter().enumerate() {
            if q > max_q {
                max_q = q;
                best_action = i;
            }
        }
        best_action
    }

    pub fn train_step(&mut self) -> f32 {
        if self.replay_buffer.len() < BATCH_SIZE {
            return 0.0
        }

        let batch = self.replay_buffer.sample(BATCH_SIZE);
        let mut total_loss = 0.0;
        for transition in batch {
            let cache = self.network.forward_with_cache(&transition.state);
            // Bellman equation: Q(s, a) = r + γ * max_a' Q_target(s', a')
            let target_val = if transition.done {
                transition.reward
            } else {
                let next_q_values = self.target_network.forward(&transition.next_state);
                let max_next_q = next_q_values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                (transition.reward + GAMMA * max_next_q).clamp(-MAX_TARGET_Q, MAX_TARGET_Q)
            };

            let mut target_q = cache.q_values.clone();
            target_q[transition.action] = target_val;
            let loss = self.network.backward(&cache, &transition.state, &target_q, LEARNING_RATE);
            total_loss += loss;
        }
        self.training_steps += 1;
        if self.training_steps % TARGET_UPDATE_INTERVAL == 0 {
            self.target_network = self.network.clone();
        }
        total_loss / BATCH_SIZE as f32
    }

    pub fn decay_epsilon(&mut self) {
        self.epsilon = (self.epsilon * self.epsilon_decay).max(self.min_epsilon);
        print!("Decaying epsilon: {:.4}\n", self.epsilon);
    }
}
