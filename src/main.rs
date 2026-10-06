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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (model_path, train, epsilon) = parse_args()?;
    let mode = if train || model_path.is_none() {
        AgentMode::Training
    } else {
        AgentMode::Inference
    };

    println!(
        "\x1b[1;34mLoading {} model in {} mode\x1b[0m",
        model_path.as_deref().unwrap_or("new"),
        match mode {
            AgentMode::Training => "training",
            AgentMode::Inference => "inference",
        }
    );

    let mut agent = match model_path {
        Some(path) => {
            Agent::load_model(&path, mode)?
        }
        None => Agent::new(mode),
    };

    if let Some(epsilon) = epsilon {
        let trainer = agent
            .trainer
            .as_mut()
            .ok_or("--epsilon can only be used in training mode")?;
        trainer.set_epsilon(epsilon);
    }

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
        .insert_resource(agent)
        .insert_resource(ParkingGoal { position: Vec2::new(300.0, 200.0) })
        .add_systems(Startup, (setup, spawn_map))
        .add_systems(Update, (agent_loop_system, (update_sensors_and_hud, update_training_hud)).chain())
        .run();

    Ok(())
}

fn parse_args() -> Result<(Option<String>, bool, Option<f32>), String> {
    let mut args = std::env::args().skip(1);
    let mut model_path = None;
    let mut train = false;
    let mut epsilon = None;

    while let Some(arg) = args.next() {
        if let Some(path) = arg.strip_prefix("--model=") {
            if path.is_empty() {
                return Err("The --model option requires a file path".to_string());
            }
            model_path = Some(path.to_string());
        } else if arg == "--model" {
            let path = args
                .next()
                .ok_or_else(|| "The --model option requires a file path".to_string())?;
            if path.is_empty() {
                return Err("The --model option requires a file path".to_string());
            }
            model_path = Some(path);
        } else if arg == "--train" {
            train = true;
        } else if let Some(epsilon_str) = arg.strip_prefix("--epsilon=") {
            let parsed_epsilon: f32 = epsilon_str
                .parse()
                .map_err(|_| "Invalid value for --epsilon. Must be a float between 0 and 1.".to_string())?;
            if !(0.0..=1.0).contains(&parsed_epsilon) {
                return Err("Invalid value for --epsilon. Must be a float between 0 and 1.".to_string());
            }
            epsilon = Some(parsed_epsilon);
        } else if arg == "--epsilon" {
            let epsilon_str = args
                .next()
                .ok_or_else(|| "The --epsilon option requires a value between 0 and 1".to_string())?;
            let parsed_epsilon: f32 = epsilon_str
                .parse()
                .map_err(|_| "Invalid value for --epsilon. Must be a float between 0 and 1.".to_string())?;
            if !(0.0..=1.0).contains(&parsed_epsilon) {
                return Err("Invalid value for --epsilon. Must be a float between 0 and 1.".to_string());
            }
            epsilon = Some(parsed_epsilon);
        } else {
            return Err(format!("Unknown argument: {arg}"));
        }
    }

    Ok((model_path, train, epsilon))
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    goal: Res<ParkingGoal>,
    agent: Res<Agent>,
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

    if agent.mode == AgentMode::Training {
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
    }

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