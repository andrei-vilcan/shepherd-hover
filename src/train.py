import os
import math
import pickle
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.random import PRNGKey
import optax
from tqdm import tqdm

from rocket_hovering_env import RocketHoverer, EnvParams as HoveringEnvParams
from vis import visualize_trajectory
from visualize import plot_training_curves


plots_dir = os.path.join(os.path.dirname(__file__), 'plots')
models_dir = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

hovering_model_path = os.path.join(models_dir, 'hovering_model.pkl')


print_level = 1


### helpers


def difficulty_coefficient(x: float) -> float:
    """f(x) = 1 / (1 + e^(-e^e * (x - 1/2)))"""
    return 1.0 / (1.0 + math.exp(-(math.e ** math.e) * (x - 0.5)))


def initialize_mlp(layer_sizes, key, scale=1e-2):
    """
    Inputs:
        layer_sizes (tuple) Tuple of shapes of the neural network layers. Includes the input shape, hidden layer shape, and output layer shape.
        key (PRNGKey)
        scale (float) standard deviation of initial weights and biases

    Return:
        params (List) Tuple of weights and biases - [ (weights_1, biases_1), ..., (weights_n, biases_n) ]
    """
    keys = jr.split(key, 2 * len(layer_sizes))
    params = []

    for i in range(len(layer_sizes[:-1])):
        input_dim, output_dim = layer_sizes[i], layer_sizes[i + 1]
        W = jr.normal(keys[2 * i], (input_dim, output_dim)) * scale
        b = jnp.zeros(output_dim)
        params.append((W, b))

    return params


def mlp_forward(params, x, activation=jax.nn.relu):
    """Forward pass through MLP."""
    for w, b in params[:-1]:
        x = activation(x @ w + b)
    w, b = params[-1]
    return x @ w + b


### policy network (actor)


def policy(params, x):
    """
    Standard MLP that predicts continuous action from Gaussian policy.

    Inputs:
        params: Parameters of the policy network, represented as PyTree.
        x: (D,) input state, where D is the dimensionality of the state observation.
    """
    mean = mlp_forward(params['mean'], x, activation=jax.nn.tanh)
    mean = jnp.tanh(mean)
    log_std = jnp.clip(params['log_std'], -2.0, 0.5)
    std = jnp.exp(log_std)

    return mean, std


def init_policy_params(key, obs_dim, action_dim, hidden_sizes=[256, 128]):
    """Initialize policy network parameters."""
    key1, key2 = jr.split(key)
    mean_layers = [obs_dim] + hidden_sizes + [action_dim]
    mean_params = initialize_mlp(mean_layers, key1)
    log_std = jnp.zeros(action_dim)
    return {'mean': mean_params, 'log_std': log_std}


def get_action(params, x, key):
    """
    Sample continuous action from Gaussian policy.

    Inputs:
        params: (PyTree) policy network parameters
        x: (D,) observation
        key: PRNGKey
    Returns:
        action: (M,) actions generated according to params, where M is the dimensionality of actions we carry out.
        mean: (M,) means of distributions of actions generated according to params.
        std: (M,) stds of distributions of actions generated according to params.
    """
    mean, std = policy(params, x)
    noise = jr.normal(key, shape=mean.shape)
    action = jnp.clip(mean + std * noise, -1.0, 1.0)
    return action, mean, std


def get_log_prob(mean, std, action):
    """
    Return the log probability of the action executed by the MLP.

    Returns:
        log probability
    """
    var = std ** 2
    return -0.5 * jnp.sum(jnp.log(2 * jnp.pi * var) + (action - mean) ** 2 / var)


### value network (critic)


def init_value_params(key, obs_dim, hidden_sizes=[256, 128]):
    """Initialize value network parameters."""
    layers = [obs_dim] + hidden_sizes + [1]
    return initialize_mlp(layers, key)


def value(params, x):
    """
    Standard MLP that returns value estimate.

    Inputs:
        params: Parameters of the policy network, represented as PyTree
        x: (D,) input state, where D is the dimensionality of the state observation.
    """
    return mlp_forward(params, x, activation=jax.nn.tanh).squeeze(-1)


### model utilities


def save_model(policy_params, value_params, filepath: str):
    """Pickle policy and value network parameters."""
    def to_numpy(params):
        if isinstance(params, dict):
            return {k: to_numpy(v) for k, v in params.items()}
        elif isinstance(params, (list, tuple)):
            return type(params)(to_numpy(p) for p in params)
        elif hasattr(params, 'tolist'):
            return params.tolist()
        else:
            return params

    with open(filepath, 'wb') as f:
        pickle.dump({
            'policy_params': to_numpy(policy_params),
            'value_params': to_numpy(value_params),
        }, f)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    """Unpickle policy and value network parameters."""
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        def to_jax(params):
            if isinstance(params, dict):
                return {k: to_jax(v) for k, v in params.items()}
            elif isinstance(params, list):
                if len(params) > 0 and isinstance(params[0], (list, tuple)) and len(params[0]) == 2:
                    return [(jnp.array(w), jnp.array(b)) for w, b in params]
                else:
                    return jnp.array(params)
            elif isinstance(params, tuple):
                return tuple(to_jax(p) for p in params)
            else:
                return jnp.array(params)

        print(f"Model loaded from {filepath}")
        return to_jax(data['policy_params']), to_jax(data['value_params'])
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None


## Intermezzo


def make_rollout_fn(env, steps_in_episode):
    """
    Generate rollouts of the rocket in parallel with Gymnax.
    The following code generates batched rollouts of the environment in parallel.
    """

    def rollout(policy_params, env_params, key):
        """Rollout a jitted gymnax episode with lax.scan."""
        key_reset, key_episode = jr.split(key)
        obs, state = env.reset_env(key_reset, env_params)

        def policy_step(carry, _):
            """lax.scan compatible step transition in jax env."""
            obs, state, rng, done = carry
            active = ~done
            rng, rng_action, rng_step = jr.split(rng, 3)
            action, mean, std = get_action(policy_params, obs, rng_action)
            next_obs, next_state, reward, step_done, _ = env.step_env(
                rng_step, state, action, env_params
            )

            reward = jnp.where(done, 0.0, reward)
            next_obs = jnp.where(done, obs, next_obs)
            new_done = done | step_done

            carry = [next_obs, next_state, rng, new_done]
            return carry, (obs, action, reward, active)

        # Scan over episode step loop
        final_carry, (obs_seq, action_seq, reward_seq, active_seq) = jax.lax.scan(
            policy_step,
            [obs, state, key_episode, jnp.array(False)],
            (),
            length=steps_in_episode
        )
        final_obs = final_carry[0]
        return obs_seq, action_seq, reward_seq, active_seq, final_obs

    @jax.jit
    def batched_rollout(policy_params, env_params, keys):
        """Batched rollout via vmap."""
        return jax.vmap(rollout, in_axes=(None, None, 0))(
            policy_params, env_params, keys
        )

    return batched_rollout


def collect_trajectory_for_viz(key, env, env_params, policy_params, max_steps):
    key, reset_key = jr.split(key)
    obs, state = env.reset_env(reset_key, env_params)
    states = [state]
    total_reward = 0.0

    for _ in range(max_steps):
        key, action_key, step_key = jr.split(key, 3)
        action, _, _ = get_action(policy_params, obs, action_key)
        obs, state, reward, done, _ = env.step_env(step_key, state, action, env_params)
        states.append(state)
        total_reward += float(reward)
        if done:
            break

    return states, total_reward, state


## GAE and loss definitions


def compute_gae(reward, curr_value, active, gamma, _lambda):
    """
    Compute Generalized Advantage Estimation (GAE).

    Inputs:
        reward: (T,) rewards per timestep
        value:  (T,) value estimates V(s_t)
        active: (T,) boolean mask for valid steps
        gamma:  discount factor
        _lambda:    GAE lambda (0 = TD(0), 1 = Monte Carlo)
    Returns:
        advantages: (T,) GAE advantage estimates
    """
    # V(s_{t+1}): shift values forward, zero-pad the last entry
    next_value = jnp.concatenate([curr_value[1:], jnp.array([0.0])])
    next_active = jnp.concatenate([active[1:], jnp.array([False])])
    next_value = jnp.where(next_active, next_value, 0.0)

    # TD residuals: δ_t = r_t + γ V(s_{t+1}) - V(s_t)
    delta = jnp.where(active, reward + gamma * next_value - curr_value, 0.0)

    # backward scan: A_t = δ_t + γλ A_{t+1}
    def gae_step(gae, inp):
        _delta, active = inp
        gae = jnp.where(active, _delta + gamma * _lambda * gae, 0.0)
        return gae, gae

    _, advantages = jax.lax.scan(gae_step, 0.0, (delta[::-1], active[::-1]))
    return advantages[::-1]


### loss functions


def loss_REINFORCE(policy_params, obs, action, advantages, active):
    """
    Compute the error term using the REINFORCE algorithm.

    Inputs:
        policy_params: (PyTree) Current parameters of the policy network
        obs: (n_batches, T, obs_dim) Batch of observations
        action: (n_bathes, T, action_dim) actions taken
        advantages: (n_batches, T) advantages
        active: (n_batches, T) mask of active timesteps

    Return:
        Error term of the parameters
    """
    def single_trajectory_loss(obs_s, action_s, adv_s, active_s):
        def step_loss(obs, action, adv, active):
            mean, std = policy(policy_params, obs)
            log_prob = get_log_prob(mean, std, action)
            return jnp.where(active, -log_prob * adv, 0.0)

        losses = jax.vmap(step_loss)(obs_s, action_s, adv_s, active_s)
        return jnp.sum(losses) / (jnp.sum(active_s) + 1e-8)

    batch_losses = jax.vmap(single_trajectory_loss)(obs, action, advantages, active)
    return jnp.mean(batch_losses)


def value_loss(value_params, obs, returns, active):
    def single_loss(obs_t, ret_t, active_t):
        v = value(value_params, obs_t)
        return jnp.where(active_t, (v - ret_t) ** 2, 0.0)

    losses = jax.vmap(jax.vmap(single_loss))(obs, returns, active)
    return jnp.sum(losses) / (jnp.sum(active) + 1e-8)


## training loop


def train_reinforce(
    seed=42,
    curriculum=None,
    gamma=0.99,
    gae_lambda=0.95,
    policy_lr=3e-4,
    value_lr=1e-3,
    n_batches=16,
    hidden_sizes=[256, 128],
    visualize_every=100
):
    """
    Train policy using REINFORCE with learned baseline.
    """

    total_episodes = sum(stage[1] for stage in curriculum)

    if print_level >= 1:
        print(f"\nRocket Hovering Training")
        print(f"\nCurriculum Summary: ({len(curriculum)} stages, {total_episodes} episodes)")
    for i, stage in enumerate(curriculum):
        _, n_eps, max_h, use_s, coeff = stage
        s_str = "sigmoid scaling" if use_s else f"fixed coeff={coeff}"
        if print_level >= 1:
            print(f"  Stage {i+1}: Hovering - {n_eps} eps, perturbation=±{max_h}m ({s_str})")

    # normalization constant calculation
    max_offset = max(stage[2] for stage in curriculum)

    norm_x = max(max_offset * 1.5, 50.0)
    norm_y = max(max_offset * 1.5, 50.0)
    norm_v = max(max_offset * 0.5, 10.0)
    norm_omega = 3.0

    if print_level >= 1:
        print(f"\nNormalization constants: norm_x={norm_x:.1f}m, norm_y={norm_y:.1f}m, "
          f"norm_v={norm_v:.1f}m/s, norm_omega={norm_omega:.1f}rad/s")

    # initialize environment
    hovering_env = RocketHoverer()

    hovering_base_params = HoveringEnvParams(norm_x=norm_x, norm_y=norm_y, norm_v=norm_v, norm_omega=norm_omega)

    obs_dim = 11
    action_dim = 4
    max_steps = hovering_base_params.max_steps_in_episode

    key = PRNGKey(seed)
    key, policy_key, value_key = jr.split(key, 3)
    policy_params = init_policy_params(policy_key, obs_dim, action_dim, hidden_sizes)
    value_params = init_value_params(value_key, obs_dim, hidden_sizes)

    policy_optimizer = optax.adam(policy_lr)
    value_optimizer = optax.adam(value_lr)
    policy_opt_state = policy_optimizer.init(policy_params)
    value_opt_state = value_optimizer.init(value_params)

    policy_grad_fn = jax.jit(jax.grad(loss_REINFORCE))
    value_grad_fn = jax.jit(jax.grad(value_loss))

    all_rewards = []
    all_lengths = []
    stage_iterations = []

    global_iter = 0

    env = hovering_env
    base_params = hovering_base_params
    batched_rollout = make_rollout_fn(env, max_steps)

    for stage_idx, stage in enumerate(curriculum):
        _, num_episodes, max_horiz, use_scaling, fixed_coeff = stage
        num_iters = max(1, num_episodes // n_batches)
        stage_iterations.append(num_iters)

        scaling_str = "difficulty scaling" if use_scaling else f"fixed coeff={fixed_coeff}"
        if print_level >= 1:
            print(f"Hovering Stage {stage_idx+1}/{len(curriculum)}: "
                  f"perturbation=±{max_horiz}m ({scaling_str})")
            print(f"  {num_episodes} episodes = {num_iters} iterations x {n_batches} batch")

        pbar = tqdm(range(num_iters), desc=f"Hovering {stage_idx+1}", unit="iter", mininterval=2.0)

        for iteration in pbar:
            global_iter += 1
            key, iter_key = jr.split(key)

            # difficulty coefficient
            progress = iteration / num_iters
            coeff = fixed_coeff
            if use_scaling:
                coeff *= difficulty_coefficient(progress)

            env_params = base_params.replace(
                init_max_horizontal_offset=max_horiz * coeff,
                init_max_vertical_offset=max_horiz * coeff,
                init_max_horizontal_velocity=base_params.init_max_horizontal_velocity * coeff,
                init_max_vertical_velocity=base_params.init_max_vertical_velocity * coeff,
                init_max_angle=base_params.init_max_angle * coeff,
                init_max_angular_velocity=base_params.init_max_angular_velocity * coeff,
            )

            # batched rollout
            keys = jr.split(iter_key, n_batches)
            obs, action, reward, active, final_obs = batched_rollout(policy_params, env_params, keys)

            # compute values for baseline
            values = jax.vmap(jax.vmap(lambda o: value(value_params, o)))(obs)

            # GAE advantages
            advantages = jax.vmap(compute_gae, in_axes=(0, 0, 0, None, None))(
                reward, values, active, gamma, gae_lambda
            )

            # estimate return
            returns = advantages + values

            adv_mean = jnp.sum(advantages * active) / (jnp.sum(active) + 1e-8)
            adv_std = jnp.sqrt(
                jnp.sum((advantages - adv_mean) ** 2 * active) / (jnp.sum(active) + 1e-8) + 1e-8
            )
            advantages = (advantages - adv_mean) / adv_std

            # update policy params
            p_grads = policy_grad_fn(policy_params, obs, action, advantages, active)
            p_updates, policy_opt_state = policy_optimizer.update(p_grads, policy_opt_state)
            policy_params = optax.apply_updates(policy_params, p_updates)

            # update value params
            v_grads = value_grad_fn(value_params, obs, returns, active)
            v_updates, value_opt_state = value_optimizer.update(v_grads, value_opt_state)
            value_params = optax.apply_updates(value_params, v_updates)

            # metrics
            mean_reward = float(jnp.mean(jnp.sum(reward, axis=-1)))
            mean_length = float(jnp.mean(jnp.sum(active, axis=-1)))
            all_rewards.append(mean_reward)
            all_lengths.append(mean_length)

            if iteration % 20 == 0:
                pbar.set_postfix({'reward': f'{mean_reward:.1f}', 'ep_len': f'{mean_length:.0f}'})

            # visualization
            if visualize_every > 0 and global_iter % visualize_every == 0:
                key, viz_key = jr.split(key)
                states, viz_reward, _ = collect_trajectory_for_viz(
                    viz_key, env, env_params, policy_params, max_steps
                )
                visualize_trajectory(
                    states, env_params,
                    episode=global_iter * n_batches,
                    reward=viz_reward,
                    filename=f'trajectory_ep{global_iter * n_batches}.png',
                )

        pbar.close()

    save_model(policy_params, value_params, hovering_model_path)

    plot_training_curves(all_rewards, all_lengths, stage_iterations, window=50)

    def print_average_reward():
        start = 0
        for i, n in enumerate(stage_iterations):
            end = start + n
            stage_rewards = all_rewards[start:end]
            avg = sum(stage_rewards) / len(stage_rewards) if stage_rewards else 0
            print(f"Stage {i+1}: average reward = {avg:.1f}")
            start = end

    print_average_reward()

    return policy_params, value_params, all_rewards


# main


if __name__ == "__main__":
    # [type, n_episodes, max_perturbation, use_scaling, difficulty_coefficient]
    curriculum = [
        ("hovering", 100000, 0.0, 0, 0),
        ("hovering", 100000, 20.0, 1, 1.0),
        ("hovering", 100000, 20.0, 0, 1.0),
    ]

    policy_params, value_params, rewards = train_reinforce(
        seed=42,
        curriculum=curriculum,
        gamma=0.99,
        gae_lambda=0.95,
        # gae_lambda=1.0,
        policy_lr=3e-4,
        value_lr=1e-3,
        n_batches=16,
        hidden_sizes=[256, 128],
        visualize_every=100
    )
