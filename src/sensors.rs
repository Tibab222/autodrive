use crate::{
    agent::{Agent, AgentMode},
    components::{Car, HudText, LossHudText, Sensor, Wall},
};
use bevy::{prelude::*, window::PrimaryWindow};

pub fn update_sensors_and_hud(
    mut car_query: Query<(&Transform, &mut Sensor), With<Car>>,
    wall_query: Query<(&Transform, &Wall), Without<HudText>>,
    mut gizmos: Gizmos,
    window_query: Query<&Window, With<PrimaryWindow>>,
    mut text_query: Query<(&mut Text2d, &mut Transform), (With<HudText>, Without<Car>)>,
) {
    let Ok((car_transform, mut sensor)) = car_query.single_mut() else {
        return;
    };
    let Ok(window) = window_query.single() else {
        return;
    };
    let half_size = Vec2::new(window.width(), window.height()) / 2.0;
    let margin = 24.0;

    let car_pos = car_transform.translation.truncate();
    let car_rotation =
        car_transform.rotation.to_euler(EulerRot::ZYX).0 + std::f32::consts::FRAC_PI_2;

    let ray_angles = sensor.ray_angles.clone();
    let num_rays = ray_angles.len();
    sensor.distances = vec![sensor.max_distance; num_rays];

    let mut hud_string = String::from("--- SENSOR RADAR ---\n");

    for (i, &angle_offset) in ray_angles.iter().enumerate() {
        let ray_angle = car_rotation + angle_offset;
        let ray_dir = Vec2::new(ray_angle.cos(), ray_angle.sin());
        let mut min_dist = sensor.max_distance;

        // collision detection with walls
        for (wall_transform, wall) in wall_query.iter() {
            let wall_pos = wall_transform.translation.truncate();
            if let Some(dist) = ray_aabb_intersection(car_pos, ray_dir, wall_pos, wall.size) {
                if dist < min_dist {
                    min_dist = dist;
                }
            }
        }

        sensor.distances[i] = min_dist;

        // overlay the ray on the main view
        let hit_point = car_pos + ray_dir * min_dist;
        let color = get_sensor_color(min_dist, sensor.max_distance);
        gizmos.line_2d(car_pos, hit_point, color);

        let label = match i {
            0 => "Left",
            1 => "Front-left",
            2 => "Front",
            3 => "Front-right",
            4 => "Right",
            5 => "Rear",
            _ => "Sensor",
        };
        hud_string.push_str(&format!("{}: {:.0} px\n", label, min_dist));
    }

    let hud_center = Vec2::new(-half_size.x + margin + 100.0, half_size.y - margin - 90.0);

    gizmos.rect_2d(
        hud_center,
        Vec2::new(20.0, 35.0),
        Color::srgb(0.3, 0.7, 1.0),
    );
    gizmos.line_2d(
        hud_center,
        hud_center + Vec2::new(0.0, 18.0),
        Color::srgb(1.0, 0.9, 0.0),
    );

    for (i, &angle_offset) in ray_angles.iter().enumerate() {
        let dist = sensor.distances[i];
        let normalized_dist = (dist / sensor.max_distance) * 80.0; // Scalé pour le HUD
        let color = get_sensor_color(dist, sensor.max_distance);

        let hud_angle = std::f32::consts::FRAC_PI_2 + angle_offset;
        let hud_dir = Vec2::new(hud_angle.cos(), hud_angle.sin());

        gizmos.line_2d(hud_center, hud_center + hud_dir * normalized_dist, color);
    }

    if let Ok((mut text, mut transform)) = text_query.single_mut() {
        text.0 = hud_string;
        transform.translation = Vec3::new(
            -half_size.x + margin + 95.0,
            half_size.y - margin - 20.0,
            10.0,
        );
    }
}

pub fn update_agent_hud(
    agent: Res<Agent>,
    mut gizmos: Gizmos,
    window_query: Query<&Window, With<PrimaryWindow>>,
    mut loss_text_query: Query<(&mut Text2d, &mut Transform), With<LossHudText>>,
) {
    let Ok(window) = window_query.single() else {
        return;
    };
    let half_size = Vec2::new(window.width(), window.height()) / 2.0;
    let margin = 24.0;

    if let Ok((mut text, mut transform)) = loss_text_query.single_mut() {
        let success_rate = if agent.recent_episode_results.is_empty() {
            0.0
        } else {
            agent
                .recent_episode_results
                .iter()
                .filter(|&&successful| successful)
                .count() as f32
                / agent.recent_episode_results.len() as f32
                * 100.0
        };
        text.0 = match agent.mode {
            AgentMode::Training => {
                let epsilon = agent.trainer.as_ref().map_or(0.0, |trainer| trainer.epsilon);
                format!(
                    "TRAINING\nEpisode: {}\nEpsilon: {:.4}\nLoss: {:.5}\n\
                     Episode reward: {:.2}\nAverage reward: {:.2}\n\
                     Steps: {}\nAverage steps: {:.1}\n\
                     Success rate: {:.1}%\nCrashes: {}\n\
                     Average distance: {:.1}\nMax Q: {:.2}",
                    agent.epoch, epsilon, agent.last_loss, agent.episode_reward,
                    agent.average_episode_reward, agent.episode_steps,
                    agent.average_episode_steps, success_rate, agent.crashed_episodes,
                    agent.average_distance, agent.last_max_q,
                )
            }
            AgentMode::Inference => format!(
                "INFERENCE\nEpisodes: {}\nSuccess rate: {:.1}%\n\
                 Successful: {}\nCrashes: {}\nCurrent steps: {}\nAverage steps: {:.1}\nMax Q: {:.2}",
                agent.total_episodes,
                success_rate,
                agent.successful_episodes,
                agent.crashed_episodes,
                agent.episode_steps,
                agent.average_episode_steps,
                agent.last_max_q,
            ),
        };
        transform.translation = Vec3::new(
            -half_size.x + margin + 115.0,
            -half_size.y + margin + 140.0,
            10.0,
        );
    }

    draw_loss_graph(
        &mut gizmos,
        &agent.loss_history,
        Vec2::new(-half_size.x + margin, -half_size.y + margin),
        Vec2::new(
            (half_size.x * 0.65).min(360.0),
            (half_size.y * 0.35).min(120.0),
        ),
    );
}

fn draw_loss_graph(gizmos: &mut Gizmos, losses: &[f32], origin: Vec2, size: Vec2) {
    let color = Color::srgb(1.0, 0.35, 0.2);

    gizmos.rect_2d(origin + size / 2.0, size, Color::srgb(0.35, 0.35, 0.4));
    if losses.len() < 2 {
        return;
    }

    let max_loss = losses.iter().copied().fold(0.0001_f32, f32::max);
    let step_x = size.x / (losses.len() - 1) as f32;
    for index in 1..losses.len() {
        let previous = origin
            + Vec2::new(
                (index - 1) as f32 * step_x,
                losses[index - 1] / max_loss * size.y,
            );
        let current = origin + Vec2::new(index as f32 * step_x, losses[index] / max_loss * size.y);
        gizmos.line_2d(previous, current, color);
    }
}

fn get_sensor_color(dist: f32, max_dist: f32) -> Color {
    let ratio = dist / max_dist;
    if ratio < 0.25 {
        Color::srgb(1.0, 0.1, 0.1) // Rouge
    } else if ratio < 0.6 {
        Color::srgb(1.0, 0.8, 0.1) // Jaune
    } else {
        Color::srgb(0.2, 0.9, 0.2) // Vert
    }
}

fn ray_aabb_intersection(origin: Vec2, dir: Vec2, box_pos: Vec2, box_size: Vec2) -> Option<f32> {
    let min_b = box_pos - box_size / 2.0;
    let max_b = box_pos + box_size / 2.0;

    let mut tmin = (min_b.x - origin.x) / dir.x;
    let mut tmax = (max_b.x - origin.x) / dir.x;

    if tmin > tmax {
        std::mem::swap(&mut tmin, &mut tmax);
    }

    let mut tymin = (min_b.y - origin.y) / dir.y;
    let mut tymax = (max_b.y - origin.y) / dir.y;

    if tymin > tymax {
        std::mem::swap(&mut tymin, &mut tymax);
    }

    if (tmin > tymax) || (tymin > tmax) {
        return None;
    }

    if tymin > tmin {
        tmin = tymin;
    }
    if tymax < tmax {
        tmax = tymax;
    }

    if tmin >= 0.0 {
        Some(tmin)
    } else {
        None
    }
}
