use bevy::prelude::*;
use rand::RngExt;
use crate::components::Wall;

pub fn spawn_map(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    let wall_material = materials.add(ColorMaterial::from(Color::srgb(0.8, 0.2, 0.2))); // red
    let wall_thickness = 20.0;

    // map dimensions
    let map_width = 1200.0;
    let map_height = 680.0;

    // walls for the borders map
    let borders = vec![
        // Position (x, y), dimensions (width, height)
        (Vec2::new(0.0, map_height / 2.0), Vec2::new(map_width, wall_thickness)), // Top
        (Vec2::new(0.0, -map_height / 2.0), Vec2::new(map_width, wall_thickness)), // Bottom
        (Vec2::new(-map_width / 2.0, 0.0), Vec2::new(wall_thickness, map_height)), // Left
        (Vec2::new(map_width / 2.0, 0.0), Vec2::new(wall_thickness, map_height)), // Right
    ];

    for (pos, size) in borders {
        spawn_wall(&mut commands, &mut meshes, wall_material.clone(), pos, size);
    }

    // random obstacles inside the map
    let mut rng = rand::rng();
    let num_obstacles = 12; // Number of obstacles to spawn

    for _ in 0..num_obstacles {
        let mut pos = Vec2::ZERO;

        while pos.length() < 150.0 {
            pos.x = rng.random_range(-map_width / 2.5..map_width / 2.5);
            pos.y = rng.random_range(-map_height / 2.5..map_height / 2.5);
        }

        let size = if rng.random_bool(0.5) {
            Vec2::new(rng.random_range(40.0..120.0), rng.random_range(40.0..60.0))
        } else {
            Vec2::new(rng.random_range(40.0..60.0), rng.random_range(40.0..120.0))
        };

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