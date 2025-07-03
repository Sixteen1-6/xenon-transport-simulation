Xenon Transport Simulation

An interactive molecular dynamics simulation exploring xenon gas transport through zinc membranes in aqueous environments

Overview
This project is a web-based molecular dynamics simulation that models the fascinating process of xenon gas transport through zinc membranes surrounded by water molecules. What started as a curiosity about how noble gases behave at membrane interfaces turned into a comprehensive physics simulation that brings quantum chemistry principles to life in your browser.
The simulation captures the intricate dance of molecules at the atomic scale, where xenon atoms navigate through membrane pores while water molecules create a dynamic environment around them. It's like watching molecular traffic, but with real physics governing every movement.
What Makes This Special
Unlike simplified educational simulations, this project implements actual physical chemistry principles:

Real molecular interactions using harmonic potentials for chemical bonds
Accurate electrostatic forces calculated via Coulomb's law
Quantum chemistry-derived parameters that reflect real molecular behavior
Temperature control systems that mimic laboratory conditions
Multiple transport mechanisms including adsorption, diffusion, and membrane permeation

The Physics Behind It
Molecular Dynamics Engine
The simulation uses the Velocity Verlet algorithm to solve Newton's equations of motion for every atom. This ensures that energy is conserved and trajectories are physically meaningful.
Chemical Bond Modeling
Water molecules aren't just bouncing balls - they have realistic O-H bonds modeled with harmonic potentials. The H-O-H angle is maintained at the correct 104.52°, just like in real water.
Electrostatic Interactions
Every atom carries realistic partial charges:

Oxygen atoms: -0.834 e
Hydrogen atoms: +0.417 e
Xenon atoms: -1 e

These charges interact via Coulomb's law, creating the complex electrostatic landscape that drives molecular behavior.
Temperature Control
The simulation includes a velocity rescaling thermostat that maintains constant temperature by periodically adjusting molecular velocities - just like how temperature is controlled in real molecular dynamics experiments.
Features

Interactive Controls: Adjust the number of molecules in real-time
Real-time Visualization: Watch molecules move and interact as the simulation runs
Physical Accuracy: Based on quantum chemistry calculations and established physics principles
Educational Value: Perfect for understanding molecular transport phenomena
Cross-platform: Runs in any modern web browser

How It Works

Initialization: The system sets up water molecules, xenon atoms, and the zinc membrane with realistic starting positions
Force Calculation: Every timestep, the simulation calculates forces between all atoms
Integration: Positions and velocities are updated using the Velocity Verlet algorithm
Visualization: The current state is rendered to show molecular positions and movements

The Science
This simulation models several key processes:
Xenon Adsorption
Xenon atoms can temporarily bind to membrane surfaces before either desorbing back into solution or proceeding through the membrane.
Membrane Permeation
The rate-limiting step where xenon must overcome an energy barrier to pass through membrane pores.
Diffusion
Both in the bulk water phase and within membrane channels, xenon movement is governed by concentration gradients and thermal motion.
Technical Implementation

Frontend: Pure JavaScript with HTML5 Canvas for visualization
Physics Engine: Custom implementation optimized for molecular dynamics
Integration: Velocity Verlet algorithm with adaptive timestep control
Performance: Optimized for smooth real-time simulation in web browsers
