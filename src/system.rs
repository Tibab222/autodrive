use bevy::prelude::*;
use rand::RngExt;

use crate::{agent::{Agent, AgentMode, CarAction, Transition, agent_car::AgentCar, encode_state}, components::{Car, GoalMarker, ParkingGoal, Sensor, Wall}, controls::step_car_physics, map::spawn_random_map};

pub fn agent_loop_system(
    mut agent: ResMut<Agent>,
    mut commands: Commands,
    time: Res<Time>,
    mut goal: ResMut<ParkingGoal>,
    mut query: Query<(&mut Car, &mut Transform, &Sensor, &mut AgentCar)>,
    wall_query: Query<(&Transform, &Wall), (Without<Car>, Without<GoalMarker>)>,
    wall_entities: Query<Entity, With<Wall>>,
    mut goal_marker_query: Query<&mut Transform, (With<GoalMarker>, Without<Car>)>,
    meshes: ResMut<Assets<Mesh>>,
    materials: ResMut<Assets<ColorMaterial>>,
) {
    let dt = time.delta_secs();
    let Ok((mut car, mut transform, sensor, mut agent_car)) = query.single_mut() else {
        return;
    };
    let current_state = encode_state(&car, &transform, sensor, goal.position);
    let current_dist = goal.position.distance(transform.translation.truncate());
    agent_car.update_parking_status(current_dist, car.speed);

    if let Some(prev_state) = agent_car.last_state.clone() {
        let (reward, done) = agent_car.compute_reward(current_dist);

        if agent.mode == AgentMode::Training {
            agent.episode_reward += reward;
            agent.episode_steps += 1;
            agent.episode_distance_sum += current_dist;
            let loss = if let Some(ref mut trainer) = agent.trainer {
                trainer.replay_buffer.push(Transition {
                    state: prev_state,
                    action: agent_car.last_action,
                    reward,
                    next_state: current_state.clone(),
                    done,
                });

                if done {
                    trainer.decay_epsilon();
                }
                trainer.train_step()
            } else {
                0.0
            };

            agent.last_loss = loss;
            if loss > 0.0 {
                agent.loss_history.push(loss);
                if agent.loss_history.len() > 120 {
                    agent.loss_history.remove(0);
                }
            }
        }

        if done {
            if agent.mode == AgentMode::Training {
                agent.total_episodes += 1;
                if agent_car.is_crashed {
                    agent.crashed_episodes += 1;
                } else {
                    agent.successful_episodes += 1;
                }
                agent.average_episode_reward =
                    (agent.average_episode_reward * (agent.total_episodes - 1) as f32
                        + agent.episode_reward)
                        / agent.total_episodes as f32;
                agent.average_episode_steps =
                    (agent.average_episode_steps * (agent.total_episodes - 1) as f32
                        + agent.episode_steps as f32)
                        / agent.total_episodes as f32;
                let episode_average_distance = if agent.episode_steps > 0 {
                    agent.episode_distance_sum / agent.episode_steps as f32
                } else {
                    0.0
                };
                agent.average_distance =
                    (agent.average_distance * (agent.total_episodes - 1) as f32
                        + episode_average_distance)
                        / agent.total_episodes as f32;
                let success_rate =
                    agent.successful_episodes as f32 / agent.total_episodes as f32;
                if agent.successful_episodes > 0 && success_rate > agent.best_success_rate {
                    agent.best_success_rate = success_rate;
                    std::fs::create_dir_all("checkpoints")
                        .expect("Failed to create checkpoints directory");
                    agent.save_model("checkpoints/best_model.bin");
                    info!(
                        "Saved best model at episode {} with {:.1}% success rate",
                        agent.epoch + 1,
                        success_rate * 100.0
                    );
                }
                agent.episode_reward = 0.0;
                agent.episode_steps = 0;
                agent.episode_distance_sum = 0.0;
                agent.epoch += 1;
                randomize_episode(
                    &mut commands,
                    &mut goal,
                    &wall_entities,
                    &mut goal_marker_query,
                    meshes,
                    materials,
                );
            }
            reset_car_position(
                &mut car,
                &mut transform,
                &mut agent_car,
                goal.position.length(),
            );
            return;
        }
    }

    let q_values = agent.infer(&current_state);
    agent.last_max_q = q_values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let action_idx = agent.select_action(&current_state);
    let action = CarAction::from_index(action_idx);

    let crashed = apply_car_action(
        action,
        &mut car,
        &mut *transform,
        dt,
        &wall_query,
    );

    if crashed {
        agent_car.is_crashed = true;
    }

    agent_car.last_state = Some(current_state);
    agent_car.last_action = action_idx;
    agent_car.last_distance_to_target = current_dist;
}

fn randomize_episode(
    commands: &mut Commands,
    goal: &mut ParkingGoal,
    wall_entities: &Query<Entity, With<Wall>>,
    goal_marker_query: &mut Query<&mut Transform, (With<GoalMarker>, Without<Car>)>,
    meshes: ResMut<Assets<Mesh>>,
    materials: ResMut<Assets<ColorMaterial>>,
) {
    for entity in wall_entities.iter() {
        commands.entity(entity).despawn();
    }

    let mut rng = rand::rng();
    goal.position = Vec2::new(
        rng.random_range(-450.0..450.0),
        rng.random_range(-250.0..250.0),
    );

    for mut transform in goal_marker_query.iter_mut() {
        transform.translation.x = goal.position.x;
        transform.translation.y = goal.position.y;
    }

    spawn_random_map(commands.reborrow(), meshes, materials);
}

fn apply_car_action(
    action: CarAction,
    car: &mut Car,
    transform: &mut Transform,
    dt: f32,
    wall_query: &Query<(&Transform, &Wall), (Without<Car>, Without<GoalMarker>)>,
) -> bool {
    let (accelerating, braking, turn_direction) = match action {
        CarAction::Neutral => (false, false, 0.0),
        CarAction::Forward => (true, false, 0.0),
        CarAction::ForwardLeft => (true, false, -1.0),
        CarAction::ForwardRight => (true, false, 1.0),
        CarAction::BrakeReverse => (false, true, 0.0),
    };

    let crashed = step_car_physics(car, transform, accelerating, braking, turn_direction, dt, wall_query);
    crashed
}

fn reset_car_position(car: &mut Car, transform: &mut Transform, agent_car: &mut AgentCar, current_dist: f32) {
    car.speed = 0.0;
    transform.translation = Vec3::new(0.0, 0.0, 0.0);
    transform.rotation = Quat::from_rotation_z(0.0);

    agent_car.is_crashed = false;
    agent_car.is_parked = false;
    agent_car.last_state = None;
    agent_car.last_action = 0;
    agent_car.last_distance_to_target = current_dist;
}