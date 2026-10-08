use bevy::prelude::*;

use crate::map::{MAP_HEIGHT, MAP_WIDTH};

const CRASH_PENALTY: f32 = -0.75;
pub const TIMEOUT_PENALTY: f32 = -0.0;
const PARKING_REWARD: f32 = 5.0;
const PARKING_RADIUS: f32 = 25.0;
const PARKING_SPEED_THRESHOLD: f32 = 10.0;
const STEP_PENALTY: f32 = 0.005;
const DISTANCE_REWARD_SCALE: f32 = 1.0;
const REWARD_GRID_CELL_SIZE: f32 = 10.0;
const REWARD_CLEARANCE: f32 = 10.0;
const UNREACHABLE_REWARD: f32 = -0.05;
const RECOVERY_REWARD: f32 = 0.02;

#[derive(Component)]
pub struct AgentCar {
    pub last_state: Option<Vec<f32>>,
    pub last_action: usize,
    pub last_distance_to_target: f32,
    pub is_crashed: bool,
    pub is_parked: bool,
    reward_map: Vec<f32>,
    reward_grid_width: usize,
    reward_grid_height: usize,
    last_reward_distance: f32,
}

impl AgentCar {
    pub fn new() -> Self {
        Self {
            last_state: None,
            last_action: 0,
            last_distance_to_target: f32::MAX,
            is_crashed: false,
            is_parked: false,
            reward_map: Vec::new(),
            reward_grid_width: 0,
            reward_grid_height: 0,
            last_reward_distance: f32::INFINITY,
        }
    }

    pub fn has_no_reward_map(&self) -> bool {
        self.reward_map.is_empty()
    }

    pub fn begin_episode(&mut self, goal: Vec2, walls: &[(Vec2, Vec2)], car_position: Vec2) {
        self.generate_reward_map(goal, walls);
        self.last_reward_distance = self.reward_distance_at(car_position);
        self.last_state = None;
        self.last_action = 0;
        self.last_distance_to_target = goal.distance(car_position);
        self.is_crashed = false;
        self.is_parked = false;
    }

    fn generate_reward_map(&mut self, goal: Vec2, walls: &[(Vec2, Vec2)]) {
        self.reward_grid_width = (MAP_WIDTH / REWARD_GRID_CELL_SIZE).ceil() as usize;
        self.reward_grid_height = (MAP_HEIGHT / REWARD_GRID_CELL_SIZE).ceil() as usize;
        self.reward_map = vec![f32::INFINITY; self.reward_grid_width * self.reward_grid_height];

        let goal_index = self.grid_index(goal);
        self.reward_map[goal_index] = 0.0;
        let mut open = vec![goal_index];

        while !open.is_empty() {
            let current_position = open
                .iter()
                .enumerate()
                .min_by(|(_, left), (_, right)| {
                    self.reward_map[**left]
                        .partial_cmp(&self.reward_map[**right])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(position, _)| position)
                .unwrap();
            let current_index = open.swap_remove(current_position);
            let current_distance = self.reward_map[current_index];
            let current_x = current_index % self.reward_grid_width;
            let current_y = current_index / self.reward_grid_width;

            for (neighbor_x, neighbor_y, movement_cost) in self.neighbors(current_x, current_y) {
                if !self.is_traversable(neighbor_x, neighbor_y, walls) {
                    continue;
                }
                let is_diagonal = neighbor_x != current_x && neighbor_y != current_y;
                if is_diagonal
                    && (!self.is_traversable(neighbor_x, current_y, walls)
                        || !self.is_traversable(current_x, neighbor_y, walls))
                {
                    continue;
                }

                let neighbor_index = neighbor_y * self.reward_grid_width + neighbor_x;
                let candidate_distance = current_distance + movement_cost;
                if candidate_distance < self.reward_map[neighbor_index] {
                    self.reward_map[neighbor_index] = candidate_distance;
                    open.push(neighbor_index);
                }
            }
        }
    }

    fn neighbors(&self, x: usize, y: usize) -> Vec<(usize, usize, f32)> {
        let mut neighbors = Vec::with_capacity(8);
        for delta_y in -1i32..=1 {
            for delta_x in -1i32..=1 {
                if delta_x == 0 && delta_y == 0 {
                    continue;
                }

                let neighbor_x = x as i32 + delta_x;
                let neighbor_y = y as i32 + delta_y;
                if neighbor_x < 0
                    || neighbor_y < 0
                    || neighbor_x >= self.reward_grid_width as i32
                    || neighbor_y >= self.reward_grid_height as i32
                {
                    continue;
                }

                let movement_cost = if delta_x != 0 && delta_y != 0 {
                    REWARD_GRID_CELL_SIZE * std::f32::consts::SQRT_2
                } else {
                    REWARD_GRID_CELL_SIZE
                };
                neighbors.push((neighbor_x as usize, neighbor_y as usize, movement_cost));
            }
        }
        neighbors
    }

    fn is_traversable(&self, x: usize, y: usize, walls: &[(Vec2, Vec2)]) -> bool {
        let position = self.grid_position(x, y);
        walls.iter().all(|(wall_position, wall_size)| {
            let half_size = *wall_size / 2.0 + Vec2::splat(REWARD_CLEARANCE);
            (position.x - wall_position.x).abs() > half_size.x
                || (position.y - wall_position.y).abs() > half_size.y
        })
    }

    fn grid_index(&self, position: Vec2) -> usize {
        let x = ((position.x + MAP_WIDTH / 2.0) / REWARD_GRID_CELL_SIZE)
            .floor()
            .clamp(0.0, self.reward_grid_width as f32 - 1.0) as usize;
        let y = ((position.y + MAP_HEIGHT / 2.0) / REWARD_GRID_CELL_SIZE)
            .floor()
            .clamp(0.0, self.reward_grid_height as f32 - 1.0) as usize;
        y * self.reward_grid_width + x
    }

    fn grid_position(&self, x: usize, y: usize) -> Vec2 {
        Vec2::new(
            -MAP_WIDTH / 2.0 + (x as f32 + 0.5) * REWARD_GRID_CELL_SIZE,
            -MAP_HEIGHT / 2.0 + (y as f32 + 0.5) * REWARD_GRID_CELL_SIZE,
        )
    }

    fn reward_distance_at(&self, position: Vec2) -> f32 {
        if self.reward_map.is_empty() {
            return f32::INFINITY;
        }
        self.reward_map[self.grid_index(position)]
    }

    pub fn compute_reward(&mut self, current_dist: f32, car_position: Vec2) -> (f32, bool) {
        if self.is_crashed {
            return (CRASH_PENALTY, true);
        }
        if self.is_parked {
            return (PARKING_REWARD, true);
        }

        let current_reward_distance = self.reward_distance_at(car_position);
        let (delta_dist, recovering) = match (
            self.last_reward_distance.is_finite(),
            current_reward_distance.is_finite(),
        ) {
            (true, true) => (self.last_reward_distance - current_reward_distance, false),
            (true, false) => (UNREACHABLE_REWARD, false),
            (false, true) => (RECOVERY_REWARD, true),
            (false, false) => (UNREACHABLE_REWARD, false),
        };
        self.last_reward_distance = current_reward_distance;

        let reward = if recovering {
            RECOVERY_REWARD
        } else {
            ((delta_dist * DISTANCE_REWARD_SCALE) - STEP_PENALTY).clamp(-0.05, 0.15)
        };
        self.last_distance_to_target = current_dist;
        (reward, false)
    }

    pub fn update_parking_status(&mut self, current_dist: f32, speed: f32) {
        self.is_parked = current_dist <= PARKING_RADIUS && speed.abs() <= PARKING_SPEED_THRESHOLD;
    }
}
