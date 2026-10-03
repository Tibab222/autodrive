use bevy::{math::{EulerRot, Vec2}, transform::components::Transform};

use crate::components::{Car, Sensor};

const MAP_WIDTH: f32 = 1280.0; // Width of the map
const MAP_HEIGHT: f32 = 720.0; // Height of the map

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CarAction {
    Neutral,
    Forward,
    ForwardLeft,
    ForwardRight,
    BrakeReverse,
}

/// Encodes the state of the car and its sensors into a vector of f32 values.
/// [0..=5] -> sensor distances (normalized)
/// [6] -> speed (normalized)
/// [7] -> distance to goal (normalized)
/// [8] -> angle to goal (normalized)
pub fn encode_state(car: &Car, transform: &Transform, sensor: &Sensor, goal: Vec2) -> Vec<f32> {
    let max_distance = sensor.max_distance;
    let mut state: Vec<f32> = sensor.distances.iter().map(|&d| d / max_distance).collect();

    let speed_normalized = car.speed / car.max_speed;
    state.push(speed_normalized);

    let car_pos = transform.translation.truncate();
    let distance_to_goal = goal.distance(car_pos);
    let max_goal_distance = (MAP_WIDTH.powi(2) + MAP_HEIGHT.powi(2)).sqrt();
    let distance_to_goal_normalized = (distance_to_goal / max_goal_distance).clamp(0.0, 1.0);
    state.push(distance_to_goal_normalized);

    let car_heading = transform.rotation.to_euler(EulerRot::ZYX).0 + std::f32::consts::FRAC_PI_2;
    let direction_to_goal = goal - car_pos;
    let global_angle_to_goal = direction_to_goal.y.atan2(direction_to_goal.x);

    let mut relative_angle = global_angle_to_goal - car_heading;

    while relative_angle > std::f32::consts::PI {
        relative_angle -= std::f32::consts::TAU; // TAU = 2 * PI
    }
    while relative_angle < -std::f32::consts::PI {
        relative_angle += std::f32::consts::TAU;
    }
    let angle_to_goal_normalized = relative_angle / std::f32::consts::PI;
    state.push(angle_to_goal_normalized);

    state
}

impl CarAction {
    pub fn from_index(index: usize) -> Self {
        match index {
            0 => CarAction::Neutral,
            1 => CarAction::Forward,
            2 => CarAction::ForwardLeft,
            3 => CarAction::ForwardRight,
            4 => CarAction::BrakeReverse,
            _ => CarAction::Neutral, // Default case
        }
    }
}

/// Decodes output
/// [0] -> Neutral
/// [1] -> Forward/accelerate (Z)
/// [2] -> Accelerate + left (Z + Q)
/// [3] -> Accelerate + right (Z + D)
/// [4] -> Brake/Reverse (S)
pub fn decode_action(output: &[f32]) -> CarAction {
    let max_index = output
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(index, _)| index)
        .unwrap_or(0);

    CarAction::from_index(max_index)
}