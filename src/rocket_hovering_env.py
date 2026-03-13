from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces


GRAVITY = 9.81


@struct.dataclass
class EnvState(environment.EnvState):
    # position
    x: jnp.ndarray
    y: jnp.ndarray
    theta: jnp.ndarray

    # velocity
    dx: jnp.ndarray
    dy: jnp.ndarray
    omega: jnp.ndarray

    # engine
    throttle: jnp.ndarray
    gimbal: jnp.ndarray
    left_thruster: jnp.ndarray
    right_thruster: jnp.ndarray

    # fuel
    fuel: jnp.ndarray

    # timestep
    time: jnp.ndarray


@struct.dataclass
class EnvParams(environment.EnvParams):
    dt: float = 0.02
    max_steps_in_episode: int = 1000

    # rocket
    rocket_height: float = 10.0
    rocket_mass: float = 1000.0
    rocket_moment_of_inertia: float = 10000.0

    # thrusters
    engine_distance_from_com: float = 5.0
    side_thruster_distance_from_com: float = 5.0
    
    # main thruster
    main_engine_max_thrust: float = 20000.0
    main_engine_max_gimbal: float = 0.3
    
    # side thruster
    side_thruster_max_thrust: float = 500.0
    
    # fuel
    rocket_initial_fuel: float = 500.0
    main_fuel_consumption_rate: float = 10.0
    side_fuel_consumption_rate: float = 0.5

    # init perturbation
    init_max_horizontal_offset: float = 50.0
    init_max_vertical_offset: float = 50.0
    init_max_horizontal_velocity: float = 10.0
    init_max_vertical_velocity: float = 10.0
    init_max_angle: float = 0.5
    init_max_angular_velocity: float = 0.5

    # reward coefficients
    reward_distance_factor: float = 2.0
    reward_velocity_factor: float = 1.0
    reward_angle_factor: float = 1.0
    reward_omega_factor: float = 1.0

    # bounds
    max_distance: float = 200.0

    # normalization constants
    norm_x: float = 1.0
    norm_y: float = 1.0
    norm_v: float = 1.0
    norm_omega: float = 1.0


class RocketHoverer(environment.Environment[EnvState, EnvParams]):
    """JAX/Gymnax implementation of a rocket hovering environment."""

    def __init__(self):
        super().__init__()

    @property
    def default_params(self) -> EnvParams:
        """Default environment parameters."""
        return EnvParams()

    def step_env(
            self,
            key: jax.Array,
            state: EnvState,
            action: jax.Array,
            params: EnvParams,
    ) -> Tuple[jax.Array, EnvState, jax.Array, jax.Array, dict]:
        """Environment-specific step transition."""
        # get actions
        throttle_action = jnp.clip(action[0], -1.0, 1.0)
        gimbal_action = jnp.clip(action[1], -1.0, 1.0)
        left_thruster_action = jnp.clip(action[2], -1.0, 1.0)
        right_thruster_action = jnp.clip(action[3], -1.0, 1.0)

        # mapping from distribution to control values
        throttle = (throttle_action + 1.0) / 2.0
        gimbal = gimbal_action * params.main_engine_max_gimbal
        left_thruster = (left_thruster_action + 1.0) / 2.0
        right_thruster = (right_thruster_action + 1.0) / 2.0

        # can only use as much fuel as we have
        fuel_needed = throttle * params.main_fuel_consumption_rate * params.dt + (left_thruster + right_thruster) * params.side_fuel_consumption_rate * params.dt
        fuel_coefficient = jnp.minimum(1.0, state.fuel / (fuel_needed + 1e-6))
        throttle *= fuel_coefficient
        left_thruster *= fuel_coefficient
        right_thruster *= fuel_coefficient
        
        ## deterministic step
        # gravity
        a_gravity_y = -GRAVITY
        
        ## control step
        # main engine
        thrust_angle = state.theta + gimbal
        F_main = throttle * params.main_engine_max_thrust
        
        # positional acceleration
        a_main_x = F_main * jnp.sin(thrust_angle) / params.rocket_mass
        a_main_y = F_main * jnp.cos(thrust_angle) / params.rocket_mass
        
        # angular acceleration
        tau_main = F_main * jnp.sin(gimbal) * params.engine_distance_from_com
        alpha_main = tau_main / params.rocket_moment_of_inertia
        
        # rcs
        F_left = left_thruster * params.side_thruster_max_thrust
        F_right = right_thruster * params.side_thruster_max_thrust

        # positional acceleration
        F_side_net_rocket = F_left - F_right
        a_side_x = F_side_net_rocket * jnp.cos(state.theta) / params.rocket_mass
        a_side_y = F_side_net_rocket * jnp.sin(state.theta) / params.rocket_mass

        # angular acceleration
        tau_left = F_left * params.side_thruster_distance_from_com
        tau_right = -F_right * params.side_thruster_distance_from_com
        alpha_side = (tau_left + tau_right) / params.rocket_moment_of_inertia

        ### new state calculation
        a_x = a_main_x + a_side_x
        a_y = a_gravity_y + a_main_y + a_side_y
        alpha = alpha_main + alpha_side

        dx = state.dx + a_x * params.dt
        dy = state.dy + a_y * params.dt
        omega = state.omega + alpha * params.dt
        
        x = state.x + dx * params.dt
        y = state.y + dy * params.dt
        theta = angle_normalize(state.theta + omega * params.dt)

        actual_fuel_used = (
            throttle * params.main_fuel_consumption_rate * params.dt +
            (left_thruster + right_thruster) * params.side_fuel_consumption_rate * params.dt
        )
        fuel = jnp.maximum(0.0, state.fuel - actual_fuel_used)
        
        # new state
        new_state = EnvState(
            x=x,
            y=y,
            theta=theta,
            dx=dx,
            dy=dy,
            omega=omega,
            throttle=throttle,
            gimbal=gimbal,
            left_thruster=left_thruster,
            right_thruster=right_thruster,
            fuel=fuel,
            time=state.time + params.dt
        )

        done = self.is_terminal(new_state, params)
        reward = self._compute_reward(new_state, params)
        key, obs_key = jax.random.split(key)
        obs = self.get_obs(new_state, params, obs_key)
        
        return obs, new_state, reward, done, {}

    def reset_env(
            self, key: jax.Array, params: EnvParams
    ) -> Tuple[jax.Array, EnvState]:
        """Environment-specific reset."""
        keys = jax.random.split(key, 7)

        # Sample initial perturbations (scaled by difficulty coefficient externally)
        x_init = jax.random.uniform(keys[0], minval=-params.init_max_horizontal_offset, maxval=params.init_max_horizontal_offset)
        y_init = jax.random.uniform(keys[1], minval=-params.init_max_vertical_offset, maxval=params.init_max_vertical_offset)

        dx_init = jax.random.uniform(keys[2], minval=-params.init_max_horizontal_velocity, maxval=params.init_max_horizontal_velocity)
        dy_init = jax.random.uniform(keys[3], minval=-params.init_max_vertical_velocity, maxval=params.init_max_vertical_velocity)

        theta_init = jax.random.uniform(keys[4], minval=-params.init_max_angle, maxval=params.init_max_angle)
        omega_init = jax.random.uniform(keys[5], minval=-params.init_max_angular_velocity, maxval=params.init_max_angular_velocity)
        
        state = EnvState(
            x=x_init,
            y=y_init,
            theta=theta_init,
            dx=dx_init,
            dy=dy_init,
            omega=omega_init,
            throttle=jnp.array(0.0),
            gimbal=jnp.array(0.0),
            left_thruster=jnp.array(0.0),
            right_thruster=jnp.array(0.0),
            fuel=jnp.array(params.rocket_initial_fuel),
            time=jnp.array(0.0)
        )
        
        return self.get_obs(state, params, keys[6]), state

    def get_obs(self, state: EnvState, params: EnvParams = None, key: jax.Array = None) -> jax.Array:
        """Applies observation function to state."""
        
        obs = jnp.array([
            state.x / params.norm_x,
            state.y / params.norm_y,
            state.dx / params.norm_v,
            state.dy / params.norm_v,
            state.theta / jnp.pi,
            state.omega / params.norm_omega,
            state.throttle,
            state.gimbal / params.main_engine_max_gimbal,
            state.left_thruster,
            state.right_thruster,
            state.fuel / params.rocket_initial_fuel,
        ])
        
        return obs

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Check whether state transition is terminal."""
        
        # timeout
        max_time = params.max_steps_in_episode * params.dt
        timeout = state.time >= max_time

        # out of bounds
        distance = jnp.sqrt(state.x**2 + state.y**2)
        out_of_bounds = distance > params.max_distance
        
        done = timeout | out_of_bounds
        return jnp.array(done)

    def _compute_reward(self, new_state: EnvState, params: EnvParams) -> jax.Array:
        # low distance from origin
        distance = jnp.sqrt(new_state.x**2 + new_state.y**2)
        distance_reward = params.reward_distance_factor * jnp.exp(-distance / 5.0)
        
        # low velocity
        velocity = jnp.sqrt(new_state.dx**2 + new_state.dy**2)
        velocity_reward = params.reward_velocity_factor * jnp.exp(-velocity / 5.0)

        # low angle
        angle = jnp.abs(new_state.theta)
        angle_reward = params.reward_angle_factor * jnp.exp(-angle / 0.5)

        # low angular velocity
        omega = jnp.abs(new_state.omega)
        omega_reward = params.reward_omega_factor * jnp.exp(-omega / 1.0)
        
        # total
        step_reward = distance_reward + angle_reward + velocity_reward + omega_reward
        
        return step_reward

    @property
    def name(self) -> str:
        """Environment name."""
        return "RocketHoverer"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 4

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
        """Action space of the environment."""
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=jnp.float32,
        )

    def observation_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
        """Observation space of the environment."""
        low = jnp.array([
            -1.0,   # x
            -1.0,   # y
            -1.0,   # dx
            -1.0,   # dy
            -1.0,   # theta
            -1.0,   # omega
            0.0,    # throttle
            -1.0,   # gimbal
            0.0,    # left_thruster
            0.0,    # right_thruster
            0.0,    # fuel
        ])
        high = jnp.array([
            1.0,    # x
            1.0,    # y
            1.0,    # dx
            1.0,    # dy
            1.0,    # theta
            1.0,    # omega
            1.0,    # throttle
            1.0,    # gimbal
            1.0,    # left_thruster
            1.0,    # right_thruster
            1.0,    # fuel
        ])
        return spaces.Box(low, high, shape=(11,), dtype=jnp.float32)

def angle_normalize(x: jax.Array) -> jax.Array:
    return ((x + jnp.pi) % (2 * jnp.pi)) - jnp.pi
