use bevy::prelude::*;
use crate::components::{Car, Wall};

pub fn car_control_system(
    keyboard: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut car_query: Query<(&mut Car, &mut Transform)>,
    wall_query: Query<(&Transform, &Wall), Without<Car>>,
) {
    let dt = time.delta_secs();

    for (mut car, mut transform) in car_query.iter_mut() {
        // Accelerating/Braking (Z/W or S)
        let accelerating = keyboard.pressed(KeyCode::KeyZ) || keyboard.pressed(KeyCode::KeyW);
        let braking = keyboard.pressed(KeyCode::KeyS);

        if accelerating {
            car.speed = (car.speed + car.acceleration * dt).min(car.max_speed);
        } else if braking {
            car.speed = (car.speed - car.acceleration * dt).max(-car.max_speed * 0.4);
        } else {
            // Natural deceleration (friction)
            if car.speed > 0.0 {
                car.speed = (car.speed - car.friction * dt).max(0.0);
            } else if car.speed < 0.0 {
                car.speed = (car.speed + car.friction * dt).min(0.0);
            }
        }

        // rotation (Q/A or D)
        let mut next_transform = *transform;
        if car.speed.abs() > 5.0 {
            let turn_left = keyboard.pressed(KeyCode::KeyQ) || keyboard.pressed(KeyCode::KeyA);
            let turn_right = keyboard.pressed(KeyCode::KeyD);

            let direction_multiplier = if car.speed >= 0.0 { 1.0 } else { -1.0 };

            if turn_left {
                next_transform.rotate_z(car.rotation_speed * direction_multiplier * dt);
            }
            if turn_right {
                next_transform.rotate_z(-car.rotation_speed * direction_multiplier * dt);
            }
        }

        // Movement
        let forward = next_transform.up().truncate();
        let movement = forward * car.speed * dt;
        let future_pos = next_transform.translation.truncate() + movement;

        // collisions
        let car_radius = car.size.max_element() / 2.0;
        let mut collided = false;

        for (wall_transform, wall) in wall_query.iter() {
            let wall_pos = wall_transform.translation.truncate();
            let wall_min = wall_pos - wall.size / 2.0;
            let wall_max = wall_pos + wall.size / 2.0;

            let closest_x = future_pos.x.clamp(wall_min.x, wall_max.x);
            let closest_y = future_pos.y.clamp(wall_min.y, wall_max.y);
            let closest_point = Vec2::new(closest_x, closest_y);

            let distance = future_pos.distance(closest_point);

            if distance < car_radius {
                collided = true;
                break;
            }
        }

        // check collision and update position
        if collided {
            car.speed = 0.0;
        } else {
            *transform = next_transform;
            transform.translation += movement.extend(0.0);
        }
    }
}