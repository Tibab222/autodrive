use bevy::prelude::*;

const CRASH_PENALTY: f32 = -5.0;
pub const TIMEOUT_PENALTY: f32 = -2.0;
const PARKING_REWARD: f32 = 5.0;
const PARKING_RADIUS: f32 = 25.0;
const PARKING_SPEED_THRESHOLD: f32 = 10.0;
const STEP_PENALTY: f32 = 0.005;
const DISTANCE_REWARD_SCALE: f32 = 1.0;

#[derive(Component)]
pub struct AgentCar {
    pub last_state: Option<Vec<f32>>,
    pub last_action: usize,
    pub last_distance_to_target: f32,
    pub is_crashed: bool,
    pub is_parked: bool,
}

impl AgentCar {
    pub fn new() -> Self {
        Self {
            last_state: None,
            last_action: 0,
            last_distance_to_target: f32::MAX,
            is_crashed: false,
            is_parked: false,
        }
    }

    // /// Constructs the state vector for the agent based on the car's position, target position, and raycast distances.
    // pub fn build_state(
    //     &self,
    //     car_transform: &Transform,
    //     target_transform: &Transform,
    //     raycasts: &[f32],
    // ) -> Vec<f32> {
    //     let rel_x = target_transform.translation.x - car_transform.translation.x;
    //     let rel_y = target_transform.translation.y - car_transform.translation.y;
    //     let current_dist = car_transform
    //         .translation
    //         .distance(target_transform.translation);

    //     let mut state = vec![rel_x, rel_y, car_transform.rotation.z, current_dist];
    //     state.extend_from_slice(raycasts); // Ajout des distances LiDAR/Raycast
    //     state
    // }

    /// Computes the reward based on the current distance to the target and the agent's state (crashed or parked).
    pub fn compute_reward(&mut self, current_dist: f32) -> (f32, bool) {
        if self.is_crashed {
            return (CRASH_PENALTY, true);
        }
        if self.is_parked {
            return (PARKING_REWARD, true);
        }

        let delta_dist = self.last_distance_to_target - current_dist;
        let reward = ((delta_dist * DISTANCE_REWARD_SCALE) - STEP_PENALTY).clamp(-0.05, 0.05);

        (reward, false)
    }

    pub fn update_parking_status(&mut self, current_dist: f32, speed: f32) {
        self.is_parked = current_dist <= PARKING_RADIUS && speed.abs() <= PARKING_SPEED_THRESHOLD;
    }
}
