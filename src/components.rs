use bevy::prelude::*;

#[derive(Component)]
pub struct Car {
    pub speed: f32,
    pub max_speed: f32,
    pub acceleration: f32,
    pub friction: f32,
    pub rotation_speed: f32,
    pub size: Vec2,
}

impl Default for Car {
    fn default() -> Self {
        Self {
            speed: 0.0,
            max_speed: 400.0, // max speed (in pixels/sec)
            acceleration: 300.0, // acceleration in pixels/sec²
            friction: 200.0, // friction in pixels/sec²
            rotation_speed: 3.5, // rotation speed in radians/sec
            size: Vec2::new(30.0, 50.0), // size of the car (width, height)
        }
    }
}

#[derive(Component)]
pub struct Wall {
    pub size: Vec2, // (width, height) of the wall
}

#[derive(Component)]
pub struct Sensor {
    pub ray_angles: Vec<f32>, // angles in rad
    pub max_distance: f32, // Max distance for the rays
    pub distances: Vec<f32>, // measured distances for each ray (initialized to max_distance)
}

impl Default for Sensor {
    fn default() -> Self {
        Self {
            ray_angles: vec![
                -std::f32::consts::FRAC_PI_2, // Left (-90°)
                -std::f32::consts::FRAC_PI_4, // Diag-Left (-45°)
                0.0, // front (0°)
                std::f32::consts::FRAC_PI_4, // Diag-Right (45°)
                std::f32::consts::FRAC_PI_2, // Right (90°)
                std::f32::consts::PI, // Back (180°)
            ],
            max_distance: 300.0,
            distances: vec![300.0; 6],
        }
    }
}

#[derive(Component)]
pub struct HudText;