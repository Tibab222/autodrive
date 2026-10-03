mod state;
mod model;
mod train;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentMode {
    Training,
    Inference
}

pub struct Agent {
    mode: AgentMode,
    model: model::NeuralNetwork,
}

impl Agent {
    // ***************** INIT FUNCTIONS *****************
    pub fn new(mode: AgentMode) -> Self {
        let model = model::NeuralNetwork::new_random();
        Agent { mode, model }
    }

    pub fn load_model(&mut self, path: &str, mode: AgentMode) -> Self {
        let model = model::NeuralNetwork::load_from_file(path).expect("Failed to load model from file");
        Agent { mode, model }
    }

    pub fn save_model(&self, path: &str) {
        self.model.save_to_file(path).expect("Failed to save model to file");
    }

    // ***************** INFERENCE MODE FUNCTIONS *****************
    pub fn infer(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(self.mode, AgentMode::Inference, "Agent is not in inference mode");
        self.model.forward(x)
    }
}