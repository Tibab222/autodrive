mod components;
mod controls;
mod map;
mod sensors;
mod agent;
mod system;

use bevy::{prelude::*, window::WindowResolution};
use components::{Car, GoalMarker, HudText, LossHudText, Sensor};
use map::spawn_map;
use sensors::{update_sensors_and_hud, update_training_hud};

use crate::{agent::{Agent, AgentMode, agent_car::AgentCar}, components::ParkingGoal, system::agent_loop_system};

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
        // Agent & Goal
        .insert_resource(Agent::new(AgentMode::Training)) // training mode
        .insert_resource(ParkingGoal { position: Vec2::new(300.0, 200.0) })
        .add_systems(Startup, (setup, spawn_map))
        .add_systems(Update, (agent_loop_system, (update_sensors_and_hud, update_training_hud)).chain())
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    goal: Res<ParkingGoal>,
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

    commands.spawn((
        Text2d::new("TRAINING\nEpoch: 0\nEpsilon: 1.0000\nLoss: waiting..."),
        TextFont {
            font_size: FontSize::from(18.0),
            ..default()
        },
        TextColor(Color::WHITE),
        Transform::from_xyz(-480.0, -135.0, 10.0),
        LossHudText,
    ));

    // Car mesh and material
    let car_mesh = meshes.add(Rectangle::new(30.0, 50.0)); // Largeur: 30px, Hauteur: 50px
    let car_material = materials.add(ColorMaterial::from(Color::srgb(0.2, 0.5, 1.0))); // Bleu

    let headlight_mesh = meshes.add(Rectangle::new(20.0, 8.0));
    let headlight_material = materials.add(ColorMaterial::from(Color::srgb(1.0, 0.9, 0.3))); // Jaune

    // Parking goal marker
    commands.spawn((
        Mesh2d(meshes.add(Circle::new(14.0))),
        MeshMaterial2d(materials.add(ColorMaterial::from(Color::srgb(0.1, 0.9, 0.2)))),
        Transform::from_xyz(goal.position.x, goal.position.y, 0.5),
        GoalMarker,
    ));

    // Car spawn
    commands
        .spawn((
            Mesh2d(car_mesh),
            MeshMaterial2d(car_material),
            Transform::from_xyz(0.0, 0.0, 1.0),
            Car::default(),
            Sensor::default(),
            AgentCar::new(),
        ))
        .with_children(|parent| {
            parent.spawn((
                Mesh2d(headlight_mesh),
                MeshMaterial2d(headlight_material),
                Transform::from_xyz(0.0, 22.0, 0.1), // Légèrement vers le haut local (+Y)
            ));
        });
}