# Simulatrix 2D Simulation Engine 🌀


**Author:** Dhanwanth.V
**Language:** Python 3 (Pygame)
**License:** MIT

---

## Overview

The **Ultimate 2D Simulation Engine** is a **high-performance, single-file physics engine** for 2D particle simulations. It combines speed, flexibility, and visual appeal, designed to run hundreds of entities in real time with minimal performance overhead.

This engine supports:

* Semi-implicit **Euler** and **Velocity Verlet** integration (toggleable)
* **Circle-circle collisions** with impulse resolution
* **Object pooling** for efficient management of short-lived particles
* **Spatial hash grid** broadphase for optimized collision detection
* **Spring constraints** and simple physics interactions
* **Dynamic HUD** showing FPS, gravity, pool status, and profiling stats
* Optional **grid overlay** and frame recording to PNG sequences
* Fully interactive **hotkeys** for live modifications

---

## Features

### Physics & Simulation

* Gravity (adjustable in real-time)
* Velocity-based collisions with restitution and friction
* Semi-implicit Euler & Velocity Verlet integration (toggle `V`)
* Object pooling to reduce memory churn
* Spring constraints for entity interactions

### Performance

* Spatial hash broadphase for fast collision detection
* Profiling of integration, broadphase, collision resolution, and constraints
* Optimized for hundreds of moving particles
* Minimal temporary allocations and memory-friendly design

### Interaction & Controls

| Key    | Action                             |
| ------ | ---------------------------------- |
| `G`    | Toggle grid overlay                |
| `H`    | Toggle HUD display                 |
| `V`    | Toggle integrator (Euler / Verlet) |
| `P`    | Pause / resume simulation          |
| `R`    | Start / stop frame recording       |
| `+`    | Spawn 100 new entities             |
| `-`    | Remove 100 entities                |
| `C`    | Clear all and respawn a small set  |
| `S`    | Toggle spatial hash broadphase     |
| `T`    | Toggle object pool usage           |
| `UP`   | Increase gravity                   |
| `DOWN` | Decrease gravity                   |
| `ESC`  | Quit simulation                    |

---

## Installation

1. Make sure you have **Python 3.x** installed.
2. Install **Pygame**:

```bash
pip install pygame
```

3. Clone or download this repository:

```bash
git clone https://github.com/yourusername/ultimate-2d-simulation.git
cd ultimate-2d-simulation
```

4. Run the engine:

```bash
python engine.py
```

---

## Usage Tips

* Use the **`V` key** to switch between Euler and Verlet integration for different physics behavior.
* **Gravity** can be adjusted in real-time with `UP` and `DOWN` keys for more dynamic simulations.
* **Object pool** reduces memory allocations when spawning/removing many particles rapidly.
* Press `R` to **record frames**, which will be saved in the `frames/` folder as PNG files.
* Experiment with springs, spawn bursts of particles, and watch collisions in real time.

---

## Folder Structure

```
ultimate-2d-simulation/
│
├── engine.py          # Main engine file (single-file engine)
├── frames/            # Saved frames when recording
├── README.md          # This file
└── screenshot.png     # Optional screenshot or GIF
```

---

## Recommended Enhancements

* Add more shapes (rectangles, polygons)
* Implement more constraints (rods, chains, etc.)
* Integrate UI sliders for gravity, restitution, friction, etc.
* Export simulations to video (from recorded PNG frames)

---

## License

This project is licensed under the **MIT License**. See `LICENSE` for details.

---

## Acknowledgements

* Inspired by physics engines and particle simulators.
* Built entirely in **Python + Pygame** for simplicity and accessibility.
