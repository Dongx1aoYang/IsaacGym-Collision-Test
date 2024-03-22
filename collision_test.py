import os
from isaacgym import gymapi, gymtorch
from vec_task import VecTask
from isaacgym.torch_utils import tensor_clamp
import torch

def to_torch(x, dtype=torch.float, device="cuda:0", requires_grad=False):
    if isinstance(x, torch.Tensor):
        x = x.to(dtype=dtype, device=device)
        x = x.clone().detach().requires_grad_(requires_grad)
    else:
        x = torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)
    return x

class CollisionTest(VecTask):
    def __init__(self, config, sim_device, graphics_device_id, headless):
        self.cfg = config
        self.num_fingers = self.cfg["env"]["num_fingers"]
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.num_joints = self.cfg["env"]["num_joints"]
        self._setup_states_obs_actions_dims()
        self.up_axis = "z"

        ## load default robot+environment parameters
        self.default_mass = self.cfg["env"]["default"]["mass"]
        self.default_COM = list(self.cfg["env"]["default"]["COM"])
        self.default_friction = self.cfg["env"]["default"]["friction"]
        self.default_scale = self.cfg["env"]["default"]["scale"]
        self.default_calibration_error = self.cfg["env"]["default"]["calibrationError"]
        self.default_stiffness = self.cfg["env"]["default"]["stiffness"]
        self.default_damping = self.cfg["env"]["default"]["damping"]
        self.default_velocity_limit = self.cfg["env"]["default"]["velocityLimit"]
        self.default_joint_friction = self.cfg["env"]["default"]["jointFriction"]
        self.default_effort_limit = self.cfg["env"]["default"]["effortLimit"]

        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless)
        
        if self.viewer is not None:
            cam_pos = gymapi.Vec3(0.5, 0.5, 0.3)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
            
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.action_scale = torch.tensor(self.cfg["env"]["action_scale"], device = self.device).repeat(self.num_joints)

        self.default_hand_joint_pos = to_torch(self.cfg["env"]["default_hand_joint_pos"], device=self.device)

        self.total_timesteps = 0

        self.create_tensor_views()
        self.gym.simulate(self.sim)
        self._refresh_tensors()
        
    def _setup_states_obs_actions_dims(self):
        dims = {
            "hand_joint_pos": self.num_joints, 
            "hand_joint_vel":  self.num_joints, 
            "hand_joint_target":  self.num_joints, 
            "all_contact": 3*(self.num_joints+1), # Assume only finger tips and thumbproxlink have tactile sensors (including the thumb)
            "hand_joint_torque":  self.num_joints, # Assume every joint has torque sensing
        }
        self.cfg["env"]["numStates"] = sum([dims[key] for key in self.cfg["env"]["feedbackState"]])
        self.obs_hist_len = self.cfg["env"]["obsHistoryLen"]
        self.cfg["env"]["numObservations"] = sum([dims[key] for key in self.cfg["env"]["feedbackObs"]]) * self.obs_hist_len
        self.cfg["env"]["numActions"] =  self.num_joints

    def create_sim(self):
        self.up_axis_idx = 2 if self.up_axis == "z" else 1
        super().create_sim()
        self._create_ground_plane()

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets"))
        hand_asset_file = os.path.join("hand", "single_finger.xml")

        # Set up asset options for hand
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = False
        # asset_options.thickness = 0.001
        asset_options.angular_damping = 0.1
        asset_options.vhacd_enabled = False
        asset_options.vhacd_params.resolution = 1000000

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = False

        hand_asset = self.gym.load_asset(self.sim, asset_root, hand_asset_file, asset_options)
        self.num_hand_bodies = self.gym.get_asset_rigid_body_count(hand_asset)
        self.hand_body_names = self.gym.get_asset_rigid_body_names(hand_asset)
        self.num_hand_shapes = self.gym.get_asset_rigid_shape_count(hand_asset)
        self.num_hand_dofs = self.gym.get_asset_dof_count(hand_asset)
        self.num_actuators = self.gym.get_asset_actuator_count(hand_asset)
        hand_dof_props = self.gym.get_asset_dof_properties(hand_asset) 

        self.finger_dof_lower_limits = list(self.cfg["env"]["joint_limits"]["finger_joint_lower_lim"]) * (self.num_fingers)
        self.finger_dof_upper_limits = list(self.cfg["env"]["joint_limits"]["finger_joint_upper_lim"]) * (self.num_fingers)

        self.hand_dof_lower_limits = to_torch(self.finger_dof_lower_limits, device=self.device)
        self.hand_dof_upper_limits = to_torch(self.finger_dof_upper_limits, device=self.device)

        hand_dof_props["velocity"] = self.default_velocity_limit
        hand_dof_props["effort"] = self.default_effort_limit

        for i in range(self.num_hand_dofs):
            hand_dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            # Don't know if this will work but it is what we have been using in MuJoCo
            hand_dof_props["stiffness"][i] = self.default_stiffness
            hand_dof_props["damping"][i] = self.default_damping
            hand_dof_props["armature"][i] = 0.01
            # Some more important properties
            # Important: If we attempt to learn finger-gaiting we must try velocity limits.
            assert hand_dof_props["hasLimits"][i], f"Joint {i} does not have limits"
            hand_dof_props["lower"][i] = self.hand_dof_lower_limits[i]
            hand_dof_props["upper"][i] = self.hand_dof_upper_limits[i]
            hand_dof_props["friction"][i] = self.default_joint_friction

        self.envs = []
        self.actor_handles = {"hand": []}
        self.hand_indices = []

        for i in range(num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.envs.append(env_ptr)

            hand_start_pose = gymapi.Transform()
            hand_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.04)
            hand_start_pose.r = gymapi.Quat(0.0 ,0.0 ,0.0 ,1.0)

            hand_actor = self.gym.create_actor(env_ptr, hand_asset, hand_start_pose, "hand", i ,-1, 0)
            self.gym.set_actor_dof_properties(env_ptr, hand_actor, hand_dof_props)

            self.actor_handles["hand"].append(hand_actor)

            hand_idx = self.gym.get_actor_index(env_ptr, hand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)

            self.gym.enable_actor_dof_force_sensors(env_ptr, hand_actor)

            friction = self.default_friction
            hand_props = self.gym.get_actor_rigid_shape_properties(env_ptr, hand_actor)

            for p in hand_props:
                p.friction = friction

            self.gym.set_actor_rigid_shape_properties(env_ptr, hand_actor, hand_props)

        self.rigid_body_handles = {}
        
        self.rigid_body_handles["proxLink1"] = self.gym.find_actor_rigid_body_handle(env_ptr, self.actor_handles["hand"][0], "proxLink1")
        self.rigid_body_handles["distLink1"] = self.gym.find_actor_rigid_body_handle(env_ptr, self.actor_handles["hand"][0], "distLink1")
        self.rigid_body_handles["palm"] = self.gym.find_actor_rigid_body_handle(env_ptr, self.actor_handles["hand"][0], "palm")

        num_actors_per_env = 1
        self.actor_idx = torch.arange(self.num_envs * num_actors_per_env, dtype=torch.int32, device=self.device).view(self.num_envs, -1)
        self.hand_idx = self.actor_idx[:, 0]
        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def create_tensor_views(self):
        # Create tensor views of simulation data
        root_state_tensor_ = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_state_tensor = gymtorch.wrap_tensor(root_state_tensor_).view(-1, 13)
        dof_state_ = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_).view(self.num_envs, -1, 2)

        rigid_body_state_ = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state_).view(self.num_envs, -1, 13)

        jacobian_ = self.gym.acquire_jacobian_tensor(self.sim, "hand")
        self.jacobian = gymtorch.wrap_tensor(jacobian_)

        # DOF Forces
        forces_ = self.gym.acquire_dof_force_tensor(self.sim)
        self.torques = gymtorch.wrap_tensor(forces_).view(self.num_envs, -1)

        contact_force_ = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.contact_force = gymtorch.wrap_tensor(contact_force_).view(self.num_envs, -1, 3)

        self.hand_dof_pos = self.dof_state[..., 0]
        self.hand_dof_vel = self.dof_state[..., 1]

        self.all_contact = [self.contact_force[:, self.rigid_body_handles["distLink1"], :]] \
            + [self.contact_force[:, self.rigid_body_handles["proxLink1"], :]] \
            + [self.contact_force[:, self.rigid_body_handles["palm"], :]]
        
        self.target_hand_joint_pos = self.hand_dof_pos.clone().detach()

    def _allocate_task_buffer(self, num_envs):
        # extra buffers for observe randomized params
        self.prop_hist_len = self.cfg["env"]["aim"]["propHistoryLen"]
        self.proprio_hist_buf = torch.zeros((num_envs, self.prop_hist_len, self.num_obs // self.obs_hist_len), device=self.device, dtype=torch.float)

    def _refresh_tensors(self):
        """Updates data in tensor views created in create_tensor_views(). To be used judiciously as
        it involves copying data.
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

    def pre_physics_step(self, actions: torch.Tensor):
        actions = actions.to(self.device)
        self.actions = actions
        self.target_hand_joint_pos += actions * self.action_scale
        self.target_hand_joint_pos = tensor_clamp(
                self.target_hand_joint_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits
            )
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.target_hand_joint_pos))

    def post_physics_step(self):
        self.progress_buf += 1
        self.reset_buf[:] = 0
        self._refresh_tensors()
        self.compute_reward()
        self.check_reset()
        env_idx = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_idx) > 0:
            self.reset_idx(env_idx)
        self.compute_observation()

    def compute_contact_bool(self):
        force_scalar = [torch.linalg.norm(force, dim=1) for force in self.all_contact]
        contact = [(force > 0).long() for force in force_scalar]
        contact_bool_tensor = torch.transpose(torch.stack(contact), 0, 1)
        contact_bool_tensor = torch.transpose(torch.stack(contact), 0, 1)
        return contact_bool_tensor

    def compute_observation(self):
        self.contact_bool_tensor = self.compute_contact_bool()
        feeback = {
            "hand_joint_pos": self.hand_dof_pos,
            "hand_joint_vel": self.hand_dof_vel,
            "hand_joint_target": self.target_hand_joint_pos,
            "all_contact": torch.cat(self.all_contact, dim=1),
            "hand_joint_torque": self.torques,
            "contact_bool": self.contact_bool_tensor,
        }
        states = {key: feeback[key] for key in self.cfg["env"]["feedbackState"]}
        obs = {key: feeback[key] for key in self.cfg["env"]["feedbackObs"]}
        self.states_buf = torch.cat(list(states.values()), dim=-1)

        prev_obs_buf = self.obs_buf_lag_history[:, 1:].clone()

        cur_obs_buf = torch.cat(list(obs.values()), dim=-1)
        cur_state_buf = torch.cat(list(states.values()), dim=-1)
        self.obs_buf_lag_history[:] = torch.cat([prev_obs_buf, cur_obs_buf.unsqueeze(1)], dim=1)

        # refill the initialized buffers
        at_reset_env_ids = self.at_reset_buf.nonzero(as_tuple=False).squeeze(-1)
        self.obs_buf_lag_history[at_reset_env_ids] = cur_obs_buf[at_reset_env_ids].unsqueeze(1)
        # pulls the last obs_hist_len observations from the history buffer
        t_buf = (self.obs_buf_lag_history[:, -self.obs_hist_len:].reshape(self.num_envs, -1)).clone()

        self.obs_buf[:, : t_buf.shape[1]] = t_buf
        self.at_reset_buf[at_reset_env_ids] = 0

    def reset(self):
        super().reset()
        self.obs_dict["proprio_hist"] = self.proprio_hist_buf.to(self.rl_device)
        if self.states_buf is not None:
            self.obs_dict["state"] = self.states_buf.clone().to(self.rl_device)
        return self.obs_dict
    
    def step(self, actions):
        self.total_timesteps += 1
        super().step(actions)
        self.obs_dict["proprio_hist"] = self.proprio_hist_buf.to(self.rl_device)
        if self.states_buf is not None:
            self.obs_dict["state"] = self.states_buf.clone().to(self.rl_device)
        if getattr(self, "success", None) is not None:
            self.extras["success"] = self.success
        return self.obs_dict, self.rew_buf.to(self.rl_device), self.reset_buf.to(self.rl_device), self.extras
    
    def compute_reward(self, actions=None):
        pass

    def check_reset(self):
        reset = self.reset_buf[:]
        reset = torch.where(self.progress_buf >= self.max_episode_length - 1, torch.ones_like(self.reset_buf), reset)
        self.reset_buf[:] = reset

    def sample_default_grasp(self, env_idx):
        hand_joint_pos = to_torch(self.default_hand_joint_pos, device=self.device).repeat(len(env_idx), 1)
        hand_joint_pos = tensor_clamp(hand_joint_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits)
        return hand_joint_pos, hand_joint_pos

    def reset_idx(self, env_idx):
        hand_joint_pos, target_hand_joint_pos = self.sample_default_grasp(env_idx)
        self.hand_dof_pos[env_idx, :] = hand_joint_pos
        self.hand_dof_vel[env_idx, :] = 0.0
        self.target_hand_joint_pos[env_idx, :] = target_hand_joint_pos
        target_hand_joint_pos[env_idx, :] = self.target_hand_joint_pos.clone().detach()
        target_hand_joint_vel = 0 * self.target_hand_joint_pos.clone().detach()

        hand_idx = self.hand_idx[env_idx].flatten()
        dof_state = self.dof_state.clone().detach()

        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(dof_state),
            gymtorch.unwrap_tensor(hand_idx),
            len(hand_idx),
        )
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(target_hand_joint_pos),
            gymtorch.unwrap_tensor(hand_idx),
            len(hand_idx),
        )
        self.gym.set_dof_velocity_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(target_hand_joint_vel),
            gymtorch.unwrap_tensor(hand_idx),
            len(hand_idx),
        )

        self.progress_buf[env_idx] = 0
        self.obs_buf[env_idx] = 0
        self.proprio_hist_buf[env_idx] = 0
        self.at_reset_buf[env_idx] = 1

        


