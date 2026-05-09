"""
    Starting cube form ground.
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np
import time

from isaacsim.core.api.world import World
from isaacsim.core.api.objects import DynamicCuboid


world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane(z_position=0.0)

cube = world.scene.add(
    DynamicCuboid(
        prim_path="/World/FallingCube",
        name="falling_cube",
        position=np.array([0.0, 0.0, 2.0]),
        size=0.5,
        mass=1.0,
        color=np.array([0.1, 0.4, 1.0]),
    )
)

world.reset()

try:
    while simulation_app.is_running():
        world.step(render=True)
        pos, quat = cube.get_world_pose()
        print(f"Cube final z: {pos[2]:.3}, {time.time()})")
except InterruptedError as e:
    print(f"\n Intrupted {e}")
finally:
    simulation_app.close()
