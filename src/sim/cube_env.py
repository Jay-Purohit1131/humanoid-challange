""" 
Pinch and place scene: Panda moves the cube onto a flate geen target square.

Built by subclassing robosuite's lift environment, which provides all the assets
like robot, table, cube observables and reward. Only the parts that differ are overriden.
"""

import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.environments.manipulation.lift import Lift

# Real-world sizes of 3X3 rubik's cube in meters. Robosuite box sizes are HALF-extents.
CUBE_HALF = 0.0285 # 5.7 cm cube
TARGET_HALF = 0.05 # 10 x 10 cm target square
TARGET_HALF_THICKNESS = 0.001 # 2 mm thick

# Success: TRUE = the whole cube is inside the square; FALSE = itd center must be.
STRICT_SUCCESS = False
RESTING_TOLERANCE = 0.01 # cube considered on the table within 1 cm of its resting height



class CubeToTarget(Lift):
    """Lift, with a fixed-orientation cube and a visual landing target."""

    def __init__(self, cube_x_range=(-0.03,0.03), cube_y_range=(-0.15,-0.09), target_xy=(0.0, 0.12), **kwargs):

        self.cube_x_range = cube_x_range
        self.cube_y_range = cube_y_range
        self.target_xy = np.array(target_xy, dtype=float)
        super().__init__(**kwargs)

    #---------------------------------------------------------------------scene
    def _load_model(self):

        ManipulationEnv._load_model(self)

        # Robot base at the table edge
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)


        arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        
        # Arena always gets set to zero origin
        arena.set_origin([0, 0, 0])

        # cube with blue face on top 
        self.cube = BoxObject(
            name="cube",
            size =[CUBE_HALF] * 3,
            rgba=[0.10, 0.30, 0.90, 1.0]
        )

        self.target = BoxObject(
            name="target",
            size=[TARGET_HALF,TARGET_HALF, TARGET_HALF_THICKNESS],
            rgba= [0.08, 0.71, 0.16, 1.0],
            obj_type="visual",
            joints=None,
        )

        #Cube starting position: random within the ranges with fixed orientation
        self.placement_initializer = UniformRandomSampler(
            name="CubeSampler",
            mujoco_objects=self.cube,
            x_range=self.cube_x_range,
            y_range=self.cube_y_range,
            rotation=0.0,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=0.01,
            rng=self.rng,
        )

        self.model = ManipulationTask(
                    mujoco_arena=arena,
                    mujoco_robots=[robot.robot_model for robot in self.robots],
                    mujoco_objects=[self.cube, self.target],
        )

    def _setup_references(self):
            super()._setup_references()
            self.target_body_id = self.sim.model.body_name2id(self.target.root_body)

    def _reset_internal(self):
         super()._reset_internal()
         table_top = self.table_offset[2]
         self.sim.model.body_pos[self.target_body_id] = np.array(
              [self.target_xy[0], self.target_xy[1], table_top + TARGET_HALF_THICKNESS]
        )
         self.sim.forward() 

    # ------------------------------------------------------------------ task
    @property
    def target_pos(self):
        return np.array(self.sim.model.body_pos[self.target_body_id])
 
    @property
    def cube_resting_z(self):
        return self.table_offset[2] + CUBE_HALF
 
    def _check_success(self):
        """Cube is inside the target square and resting on the table."""
        
        cube = self.sim.data.body_xpos[self.cube_body_id]
        allowed = TARGET_HALF - (CUBE_HALF if STRICT_SUCCESS else 0.0)
        inside = np.all(np.abs(cube[:2] - self.target_pos[:2]) <= allowed)
        resting = abs(cube[2] - self.cube_resting_z) <= RESTING_TOLERANCE
        return bool(inside and resting)

