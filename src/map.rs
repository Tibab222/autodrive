use bevy::prelude::*;
use rand::RngExt;
use crate::components::Wall;

pub const MAP_WIDTH: f32 = 1200.0;
pub const MAP_HEIGHT: f32 = 680.0;
pub const WALL_THICKNESS: f32 = 20.0;

#[derive(Resource)]
pub struct MapLayout {
    pub walls: Vec<(Vec2, Vec2)>,
}

impl MapLayout {
    pub fn random() -> Self {
        let mut rng = rand::rng();
        let mut walls = vec![
            (
                Vec2::new(0.0, MAP_HEIGHT / 2.0),
                Vec2::new(MAP_WIDTH, WALL_THICKNESS),
            ),
            (
                Vec2::new(0.0, -MAP_HEIGHT / 2.0),
                Vec2::new(MAP_WIDTH, WALL_THICKNESS),
            ),
            (
                Vec2::new(-MAP_WIDTH / 2.0, 0.0),
                Vec2::new(WALL_THICKNESS, MAP_HEIGHT),
            ),
            (
                Vec2::new(MAP_WIDTH / 2.0, 0.0),
                Vec2::new(WALL_THICKNESS, MAP_HEIGHT),
            ),
        ];

        for _ in 0..12 {
            let mut pos = Vec2::ZERO;
            while pos.length() < 150.0 {
                pos.x = rng.random_range(-MAP_WIDTH / 2.5..MAP_WIDTH / 2.5);
                pos.y = rng.random_range(-MAP_HEIGHT / 2.5..MAP_HEIGHT / 2.5);
            }

            let size = if rng.random_bool(0.5) {
                Vec2::new(rng.random_range(40.0..120.0), rng.random_range(40.0..60.0))
            } else {
                Vec2::new(rng.random_range(40.0..60.0), rng.random_range(40.0..120.0))
            };
            walls.push((pos, size));
        }

        Self { walls }
    }
}

pub fn random_valid_goal(walls: &[(Vec2, Vec2)]) -> Vec2 {
    let mut rng = rand::rng();
    let goal_radius = 20.0;
    let car_half_size = Vec2::new(15.0, 25.0);
    let min_x = -MAP_WIDTH / 2.0 + WALL_THICKNESS + goal_radius;
    let max_x = MAP_WIDTH / 2.0 - WALL_THICKNESS - goal_radius;
    let min_y = -MAP_HEIGHT / 2.0 + WALL_THICKNESS + goal_radius;
    let max_y = MAP_HEIGHT / 2.0 - WALL_THICKNESS - goal_radius;

    for _ in 0..10_000 {
        let candidate = Vec2::new(rng.random_range(min_x..max_x), rng.random_range(min_y..max_y));
        if candidate.distance(Vec2::ZERO) <= car_half_size.length() + goal_radius {
            continue;
        }

        let overlaps_wall = walls.iter().any(|(wall_pos, wall_size)| {
            let half_size = *wall_size / 2.0 + Vec2::splat(goal_radius);
            (candidate.x - wall_pos.x).abs() <= half_size.x
                && (candidate.y - wall_pos.y).abs() <= half_size.y
        });
        if !overlaps_wall {
            return candidate;
        }
    }

    panic!("Could not find a valid parking goal position");
}

pub fn spawn_map(
    commands: Commands,
    meshes: ResMut<Assets<Mesh>>,
    materials: ResMut<Assets<ColorMaterial>>,
    layout: Res<MapLayout>,
) {
    spawn_layout(commands, meshes, materials, &layout);
}

pub fn spawn_random_map(
    commands: Commands,
    meshes: ResMut<Assets<Mesh>>,
    materials: ResMut<Assets<ColorMaterial>>,
    layout: &MapLayout,
) {
    spawn_layout(commands, meshes, materials, layout);
}

fn spawn_layout(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    layout: &MapLayout,
) {
    let wall_material = materials.add(ColorMaterial::from(Color::srgb(0.8, 0.2, 0.2))); // red
    for &(pos, size) in &layout.walls {
        spawn_wall(&mut commands, &mut meshes, wall_material.clone(), pos, size);
    }
}

// helper function to spawn a wall
fn spawn_wall(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    material: Handle<ColorMaterial>,
    position: Vec2,
    size: Vec2,
) {
    let mesh = meshes.add(Rectangle::new(size.x, size.y));

    commands.spawn((
        Mesh2d(mesh),
        MeshMaterial2d(material),
        Transform::from_translation(position.extend(0.0)),
        Wall { size },
    ));
}