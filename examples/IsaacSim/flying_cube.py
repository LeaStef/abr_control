import numpy as np

class FlyingCube():

    def __init__(self, dt, world, has_collision = True, prim_path="/World/random_cube", name="fancy_cube", position=np.array([0, 0, 1.0]), scale=np.array([0.1015, 0.1015, 0.1015]), color=np.array([0, 0, 1.0])):
        # IsaacSim imports must be done after instantiating the SimulationApp, which is done by the setup in the isaacsim interface
        from omni.isaac.core.objects import DynamicCuboid # type: ignore
        from omni.isaac.dynamic_control import _dynamic_control # type: ignore

        self.world = world
        if has_collision:
            fancy_cube = self.world.scene.add(
                DynamicCuboid(
                    prim_path=prim_path,
                    name=name,
                    position=position,
                    scale=scale,
                    color=color,
                )
            )
            # Set the cube to be dynamic using dynamic_control
            self._dc = _dynamic_control.acquire_dynamic_control_interface()
            self.cube_handle = self._dc.get_rigid_body(prim_path)
            self._cube = self.world.scene.get_object(name)

            # Add velocity as an attribute of the world so that it is available at callback time
            self.input_velocity = np.zeros(3)

            self.world.add_physics_callback(name + "_send_actions", self.send_actions)
            self.world.add_physics_callback(name + "_make_state", self.make_state)
        else:
            prim_path = "/World/VisualOnlyCube"
            prim_type = "Cube"
            # Create a basic cube prim
            
            

        

    def send_actions(self, dt):
        self._dc.set_rigid_body_linear_velocity(self.cube_handle, self.input_velocity)
        
    def make_state(self, dt):
        position, orientation = self._cube.get_world_pose()
        linear_velocity = self._cube.get_linear_velocity()
        return {
            'position': np.array([[0., 0., 1.]]),
            'velocity': np.array([[0., 0., 0.]]),
            'orientation': np.array([[0., 0., 0., 0.]])
        }

    def step(self, velocity):
        # Update the input velocity
        self.input_velocity = velocity
        self.world.step()

        # Get the cube's current state
        position, orientation = self._cube.get_world_pose()
        linear_velocity = self._cube.get_linear_velocity()
        return np.concatenate([position, linear_velocity, orientation], axis=0)
    
    def _get_obj_pos(self):
        position, orientation = self._cube.get_world_pose()
        return position