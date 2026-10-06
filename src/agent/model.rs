use std::fs::File;
use std::io::{Read, Write, Result};

use rand::RngExt;

pub const INPUT_SIZE: usize = 9;
const HIDDEN_LAYER1_SIZE: usize = 32;
const HIDDEN_LAYER2_SIZE: usize = 16;
pub const OUTPUT_SIZE: usize = 5;

pub struct ForwardCache {
    pub z1: Vec<f32>,
    pub a1: Vec<f32>,
    pub z2: Vec<f32>,
    pub a2: Vec<f32>,
    pub q_values: Vec<f32>,
}

#[derive(Clone)]
pub struct NeuralNetwork {
    // Couche 1 (INPUT_SIZE -> HIDDEN_LAYER1_SIZE)
    w1: Vec<f32>, // flattened matrix [HIDDEN_LAYER1_SIZE x INPUT_SIZE]
    b1: Vec<f32>, // vector [HIDDEN_LAYER1_SIZE]

    // Couche 2 (HIDDEN_LAYER1_SIZE -> HIDDEN_LAYER2_SIZE)
    w2: Vec<f32>, // flattened matrix [HIDDEN_LAYER2_SIZE x HIDDEN_LAYER1_SIZE]
    b2: Vec<f32>, // vector [HIDDEN_LAYER2_SIZE]

    // Couche de Sortie (HIDDEN_LAYER2_SIZE -> OUTPUT_SIZE)
    w3: Vec<f32>, // flattened matrix [OUTPUT_SIZE x HIDDEN_LAYER2_SIZE]
    b3: Vec<f32>, // vector [OUTPUT_SIZE]
}

impl NeuralNetwork {
    const GRADIENT_CLIP: f32 = 1.0;

    pub fn new_random() -> Self {
        let mut rng = rand::rng();

        let w1: Vec<f32> = (0..(HIDDEN_LAYER1_SIZE * INPUT_SIZE))
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();
        let b1: Vec<f32> = (0..HIDDEN_LAYER1_SIZE)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();

        let w2: Vec<f32> = (0..(HIDDEN_LAYER2_SIZE * HIDDEN_LAYER1_SIZE))
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();
        let b2: Vec<f32> = (0..HIDDEN_LAYER2_SIZE)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();

        let w3: Vec<f32> = (0..(OUTPUT_SIZE * HIDDEN_LAYER2_SIZE))
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();
        let b3: Vec<f32> = (0..OUTPUT_SIZE)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();

        NeuralNetwork { w1, b1, w2, b2, w3, b3 }
    }

    fn relu(x: f32) -> f32 {
        x.max(0.0)
    }

    fn relu_derivative(x: f32) -> f32 {
        if x > 0.0 { 1.0 } else { 0.0 }
    }

    fn linear_layer(
        input: &[f32],
        weights: &[f32],
        biases: &[f32],
        input_size: usize,
        output_size: usize,
        apply_activation: bool, // reLu
    ) -> Vec<f32> {
        let mut output = vec![0.0; output_size];

        for i in 0..output_size {
            let mut sum = biases[i];
            let row_offset = i * input_size;
            for j in 0..input_size {
                sum += weights[row_offset + j] * input[j];
            }
            output[i] = if apply_activation { Self::relu(sum) } else { sum };
        }

        output
    }

    /// inference the neural network given an input vector
    /// reLu(wx + b) for hidden layers, and linear for output layer
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), INPUT_SIZE, "Input vector must have length {}", INPUT_SIZE);

        // first layer
        let layer1_output = Self::linear_layer(
            x,
            &self.w1,
            &self.b1,
            INPUT_SIZE,
            HIDDEN_LAYER1_SIZE,
            true,
        );

        // second layer
        let layer2_output = Self::linear_layer(
            &layer1_output,
            &self.w2,
            &self.b2,
            HIDDEN_LAYER1_SIZE,
            HIDDEN_LAYER2_SIZE,
            true,
        );

        // output layer
        let output = Self::linear_layer(
            &layer2_output,
            &self.w3,
            &self.b3,
            HIDDEN_LAYER2_SIZE,
            OUTPUT_SIZE,
            false, // no activation for output layer
        );
        output
    }

    // **********************************************************************************
    // TRAINING FUNCTIONS ***************************************************************
    // **********************************************************************************

    /// Forward pass with caching intermediate values for backpropagation
    pub fn forward_with_cache(&self, x: &[f32]) -> ForwardCache {
        assert_eq!(x.len(), INPUT_SIZE, "Input vector must have length {}", INPUT_SIZE);

        // first layer
        let z1 = Self::linear_layer(
            x,
            &self.w1,
            &self.b1,
            INPUT_SIZE,
            HIDDEN_LAYER1_SIZE,
            false, // no activation for z
        );
        let a1: Vec<f32> = z1.iter().map(|&v| Self::relu(v)).collect();

        // second layer
        let z2 = Self::linear_layer(
            &a1,
            &self.w2,
            &self.b2,
            HIDDEN_LAYER1_SIZE,
            HIDDEN_LAYER2_SIZE,
            false, // no activation for z
        );
        let a2: Vec<f32> = z2.iter().map(|&v| Self::relu(v)).collect();

        // output layer
        let q_values = Self::linear_layer(
            &a2,
            &self.w3,
            &self.b3,
            HIDDEN_LAYER2_SIZE,
            OUTPUT_SIZE,
            false, // no activation for output layer
        );

        ForwardCache { z1, a1, z2, a2, q_values }
    }

    pub fn backward(&mut self, cache: &ForwardCache, x: &[f32], target_q: &[f32], learning_rate: f32) -> f32 {
        assert_eq!(x.len(), INPUT_SIZE, "Input vector must have length {}", INPUT_SIZE);
        assert_eq!(target_q.len(), OUTPUT_SIZE, "Target Q vector must have length {}", OUTPUT_SIZE);

        let mut loss = 0.0;

        // Output layer gradients
        let mut d3 = vec![0.0f32; OUTPUT_SIZE];
        for i in 0..OUTPUT_SIZE {
            let err = cache.q_values[i] - target_q[i];
            let abs_err = err.abs();
            loss += if abs_err <= 1.0 {
                0.5 * err * err
            } else {
                abs_err - 0.5
            };
            d3[i] = err.clamp(-Self::GRADIENT_CLIP, Self::GRADIENT_CLIP);
        }
        // W3 = W3 - learning_rate * d3 * a2^T
        for i in 0..OUTPUT_SIZE {
            let row_offset = i * HIDDEN_LAYER2_SIZE;
            for j in 0..HIDDEN_LAYER2_SIZE {
                self.w3[row_offset + j] -= learning_rate * d3[i] * cache.a2[j];
            }
            self.b3[i] -= learning_rate * d3[i];
        }

        // Hidden layer 2 gradients
        // d2 = (W3^T * d3) * ReLU'(z2)
        let mut d2 = vec![0.0f32; HIDDEN_LAYER2_SIZE];
        for i in 0..HIDDEN_LAYER2_SIZE {
            let mut sum = 0.0;
            for j in 0..OUTPUT_SIZE {
                sum += self.w3[j * HIDDEN_LAYER2_SIZE + i] * d3[j];
            }
            d2[i] = (sum * Self::relu_derivative(cache.z2[i]))
                .clamp(-Self::GRADIENT_CLIP, Self::GRADIENT_CLIP);
        }
        // W2 = W2 - learning_rate * d2 * a1^T
        for i in 0..HIDDEN_LAYER2_SIZE {
            let row_offset = i * HIDDEN_LAYER1_SIZE;
            for j in 0..HIDDEN_LAYER1_SIZE {
                self.w2[row_offset + j] -= learning_rate * d2[i] * cache.a1[j];
            }
            self.b2[i] -= learning_rate * d2[i];
        }

        // Hidden layer 1 gradients
        // d1 = (W2^T * d2) * ReLU'(z1)
        let mut d1 = vec![0.0f32; HIDDEN_LAYER1_SIZE];
        for i in 0..HIDDEN_LAYER1_SIZE {
            let mut sum = 0.0;
            for j in 0..HIDDEN_LAYER2_SIZE {
                sum += self.w2[j * HIDDEN_LAYER1_SIZE + i] * d2[j];
            }
            d1[i] = (sum * Self::relu_derivative(cache.z1[i]))
                .clamp(-Self::GRADIENT_CLIP, Self::GRADIENT_CLIP);
        }
        // W1 = W1 - learning_rate * d1 * x^T
        for i in 0..HIDDEN_LAYER1_SIZE {
            let row_offset = i * INPUT_SIZE;
            for j in 0..INPUT_SIZE {
                self.w1[row_offset + j] -= learning_rate * d1[i] * x[j];
            }
            self.b1[i] -= learning_rate * d1[i];
        }

        loss / OUTPUT_SIZE as f32
    }

    // **********************************************************************************
    // Load and Save functions for the neural network weights and biases
    // **********************************************************************************

    pub fn save_to_file(&self, path: &str) -> Result<()> {
        let mut file = File::create(path)?;

        let mut write_slice = |slice: &[f32]| -> Result<()> {
            for &value in slice {
                file.write_all(&value.to_le_bytes())?;
            }
            Ok(())
        };

        write_slice(&self.w1)?;
        write_slice(&self.b1)?;
        write_slice(&self.w2)?;
        write_slice(&self.b2)?;
        write_slice(&self.w3)?;
        write_slice(&self.b3)?;

        Ok(())
    }

    pub fn load_from_file(path: &str) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let mut reader = std::io::BufReader::new(file);

        let mut read_slice = |slice: &mut [f32]| -> Result<()> {
            for i in 0..slice.len() {
                let mut bytes = [0u8; 4];
                reader.read_exact(&mut bytes)?;
                slice[i] = f32::from_le_bytes(bytes);
            }
            Ok(())
        };

        let mut w1 = vec![0.0f32; HIDDEN_LAYER1_SIZE * INPUT_SIZE];
        let mut b1 = vec![0.0f32; HIDDEN_LAYER1_SIZE];
        let mut w2 = vec![0.0f32; HIDDEN_LAYER2_SIZE * HIDDEN_LAYER1_SIZE];
        let mut b2 = vec![0.0f32; HIDDEN_LAYER2_SIZE];
        let mut w3 = vec![0.0f32; OUTPUT_SIZE * HIDDEN_LAYER2_SIZE];
        let mut b3 = vec![0.0f32; OUTPUT_SIZE];

        read_slice(&mut w1)?;
        read_slice(&mut b1)?;
        read_slice(&mut w2)?;
        read_slice(&mut b2)?;
        read_slice(&mut w3)?;
        read_slice(&mut b3)?;

        Ok(Self { w1, b1, w2, b2, w3, b3 })
    }
}