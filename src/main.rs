mod components;
mod controls;
mod map;
mod sensors;
mod agent;

use bevy::{prelude::*, window::WindowResolution};
use components::{Car, HudText, Sensor};
use controls::car_control_system;
use map::spawn_map;
use sensors::update_sensors_and_hud;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "2D Car simulation - Bevy".into(),
                resolution: WindowResolution::new(1280, 720),
                ..default()
            }),
            ..default()
        }))
        // Background
        .insert_resource(ClearColor(Color::srgb(0.1, 0.1, 0.12)))
        .add_systems(Startup, (setup, spawn_map))
        .add_systems(Update, (car_control_system, update_sensors_and_hud))
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    // 2D camera (on top)
    commands.spawn(Camera2d);

    commands.spawn((
        Text2d::new("--- SENSOR RADAR ---"),
        TextFont {
            font_size: FontSize::from(18.0),
            ..default()
        },
        TextColor(Color::WHITE),
        Transform::from_xyz(-560.0, 315.0, 10.0),
        HudText,
    ));

    // Car mesh and material
    let car_mesh = meshes.add(Rectangle::new(30.0, 50.0)); // Largeur: 30px, Hauteur: 50px
    let car_material = materials.add(ColorMaterial::from(Color::srgb(0.2, 0.5, 1.0))); // Bleu

    let headlight_mesh = meshes.add(Rectangle::new(20.0, 8.0));
    let headlight_material = materials.add(ColorMaterial::from(Color::srgb(1.0, 0.9, 0.3))); // Jaune

    // Car spawn
    commands
        .spawn((
            Mesh2d(car_mesh),
            MeshMaterial2d(car_material),
            Transform::from_xyz(0.0, 0.0, 1.0),
            Car::default(),
            Sensor::default(),
        ))
        .with_children(|parent| {
            // Petit rectangle jaune à l'avant pour savoir où est le nez de la voiture
            parent.spawn((
                Mesh2d(headlight_mesh),
                MeshMaterial2d(headlight_material),
                Transform::from_xyz(0.0, 22.0, 0.1), // Légèrement vers le haut local (+Y)
            ));
        });
}