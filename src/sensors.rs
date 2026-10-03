use bevy::prelude::*;
use crate::components::{Car, Sensor, Wall, HudText};

pub fn update_sensors_and_hud(
    mut car_query: Query<(&Transform, &mut Sensor), With<Car>>,
    wall_query: Query<(&Transform, &Wall)>,
    mut gizmos: Gizmos,
    mut text_query: Query<&mut Text, With<HudText>>,
) {
    let Ok((car_transform, mut sensor)) = car_query.single_mut() else { return; };

    let car_pos = car_transform.translation.truncate();
    let car_rotation = car_transform.rotation.to_euler(EulerRot::ZYX).0 + std::f32::consts::FRAC_PI_2;

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
            0 => "Gau",
            1 => "D-G",
            2 => "Avt",
            3 => "D-D",
            4 => "Dro",
            _ => "Ray",
        };
        hud_string.push_str(&format!("{}: {:.0} px\n", label, min_dist));
    }

    let hud_center = Vec2::new(-560.0, 280.0);
    
    gizmos.rect_2d(hud_center, Vec2::new(20.0, 35.0), Color::srgb(0.3, 0.7, 1.0));
    gizmos.line_2d(hud_center, hud_center + Vec2::new(0.0, 18.0), Color::srgb(1.0, 0.9, 0.0));

    for (i, &angle_offset) in ray_angles.iter().enumerate() {
        let dist = sensor.distances[i];
        let normalized_dist = (dist / sensor.max_distance) * 80.0; // Scalé pour le HUD
        let color = get_sensor_color(dist, sensor.max_distance);

        let hud_angle = std::f32::consts::FRAC_PI_2 + angle_offset;
        let hud_dir = Vec2::new(hud_angle.cos(), hud_angle.sin());

        gizmos.line_2d(hud_center, hud_center + hud_dir * normalized_dist, color);
    }

    if let Ok(mut text) = text_query.single_mut() {
        text.0 = hud_string;
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

    if tmin > tmax { std::mem::swap(&mut tmin, &mut tmax); }

    let mut tymin = (min_b.y - origin.y) / dir.y;
    let mut tymax = (max_b.y - origin.y) / dir.y;

    if tymin > tymax { std::mem::swap(&mut tymin, &mut tymax); }

    if (tmin > tymax) || (tymin > tmax) { return None; }

    if tymin > tmin { tmin = tymin; }
    if tymax < tmax { tmax = tymax; }

    if tmin >= 0.0 { Some(tmin) } else { None }
}