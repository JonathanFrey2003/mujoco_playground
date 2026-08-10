#!/usr/bin/env python3
import argparse
import functools
import json
import os
from datetime import datetime
from pathlib import Path
import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from jax import numpy as jnp

# Must be set before importing JAX.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_gpu_triton_gemm_any=True"
).strip()

import jax
import numpy as np
from flax import linen as nn
from gnn_implementation import ppo_networks_gnn as ppo_networks
from gnn_implementation import train as ppo
from mujoco_playground import registry, wrapper
from mujoco_playground.config import manipulation_params

ACTIVATION = {
    'celu': nn.activation.celu,
    'compact': nn.activation.compact,
    'elu': nn.activation.elu,
    'gelu': nn.activation.gelu,
    'glu': nn.activation.glu,
    'hard_sigmoid': nn.activation.hard_sigmoid,
    'hard_silu': nn.activation.hard_silu,
    'hard_swish': nn.activation.hard_swish,
    'hard_tanh': nn.activation.hard_tanh,
    'leaky_relu': nn.activation.leaky_relu,
    'linear': lambda x: x,
    'log_sigmoid': nn.activation.log_sigmoid,
    'log_softmax': nn.activation.log_softmax,
    'logsumexp': nn.activation.logsumexp,
    'mish': jax.nn.mish,
    'normalize': nn.activation.normalize,
    'one_hot': nn.activation.one_hot,
    'relu': nn.activation.relu,
    'relu6': nn.activation.relu6,
    'selu': nn.activation.selu,
    'sigmoid': nn.activation.sigmoid,
    'silu': nn.activation.silu,
    'soft_sign': nn.activation.soft_sign,
    'softmax': nn.activation.softmax,
    'softplus': nn.activation.softplus,
    'standardize': nn.activation.standardize,
    'swish': nn.activation.swish,
    'tanh': nn.activation.tanh,
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="LeapCubeReorient")
    parser.add_argument("--policy_type", choices=["abd", "mlp", "gnn"], default="abd")
    parser.add_argument("--num_timesteps", type=int, default=100_000_000)
    parser.add_argument("--num_envs", type=int, default=128)
    parser.add_argument("--num_evals", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output_dir", default="cluster_runs")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--encoder_dim", type=int, nargs="+", default=[64])
    parser.add_argument("--decoder_dim", type=int, nargs="+", default=[64])
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--activation", type=str, choices=ACTIVATION.keys(), default="relu")
    return parser.parse_args()


def main():
    args = parse_args()

    run_name = (
        f"{args.env_name}_{args.policy_type}_seed{args.seed}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir = (Path(args.output_dir) / run_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    env = registry.load(args.env_name)
    print("Devices:", jax.devices())
    print("Observation size:", env.observation_size)
    print("Action size:", env.action_size)

    ppo_params = manipulation_params.brax_ppo_config(args.env_name)
    ppo_params["num_timesteps"] = args.num_timesteps
    ppo_params["num_envs"] = args.num_envs
    ppo_params["num_evals"] = args.num_evals
    ppo_params["learning_rate"] = args.learning_rate

    ppo_training_params = dict(ppo_params)
    edges = jnp.array([[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16]])  # shape (n_edges, 2)
    senders = jnp.concatenate([edges[:, 0], edges[:,1]], axis=0)
    print("Senders:", senders)
    receivers = jnp.concatenate([edges[:, 1], edges[:,0]], axis=0)
    print("Receivers:", receivers)
    
    network_factory = ppo_networks.make_ppo_networks
    if "network_factory" in ppo_params:
        del ppo_training_params["network_factory"]
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks,
            edges=edges,
            policy_type=args.policy_type,
            encoder_dim=args.encoder_dim,
            decoder_dim=args.decoder_dim,
            hidden_dim=args.hidden_dim,
            activation=ACTIVATION[args.activation],
            **ppo_params.network_factory,
        )

    history = []

    def progress(num_steps, metrics):
        row = {"num_steps": int(num_steps)}
        for key, value in metrics.items():
            try:
                row[key] = float(np.asarray(value))
            except (TypeError, ValueError):
                row[key] = str(value)

        history.append(row)

        with open(output_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")

        print(
            f"step={num_steps} "
            f"reward={row.get('eval/episode_reward')} "
            f"std={row.get('eval/episode_reward_std')}",
            flush=True,
        )

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    ppo_training_params["save_checkpoint_path"] = str(checkpoint_dir)

    train_fn = functools.partial(
        ppo.train,
        **ppo_training_params,
        network_factory=network_factory,
        progress_fn=progress,
        seed=args.seed,
    )

    start = datetime.now()

    make_inference_fn, params, metrics = train_fn(
        environment=env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )

    # Save final parameters as a JAX/NumPy object file.
    # For long-term checkpointing, prefer your train.py save_checkpoint_path option.
    np.save(output_dir / "final_params.npy", np.asarray(params, dtype=object), allow_pickle=True)

    # ============================================================
    # Save reward plot
    # ============================================================

    steps = [
        row["num_steps"]
        for row in history
        if "eval/episode_reward" in row
    ]

    rewards = [
        row["eval/episode_reward"]
        for row in history
        if "eval/episode_reward" in row
    ]

    reward_stds = [
        row["eval/episode_reward_std"]
        for row in history
        if "eval/episode_reward" in row
    ]

    plt.figure(figsize=(10, 6))

    plt.errorbar(
        steps,
        rewards,
        yerr=reward_stds,
        capsize=3,
    )

    plt.xlabel("Environment steps")
    plt.ylabel("Mean episode reward")
    plt.title(
        f"{args.env_name} - {args.policy_type} - seed {args.seed}"
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = output_dir / "reward_curve.png"

    plt.savefig(
        plot_path,
        dpi=200,
    )

    plt.close()

    print("Saved reward plot:", plot_path)

    # ============================================================
    # Create final rollout
    # ============================================================

    NUM_VIDEOS = 128
    ROLLOUT_SEED = args.seed + 1000

    inference_fn = make_inference_fn(
        params,
        deterministic=True,
    )

    jit_inference_fn = jax.jit(inference_fn)

    infer_env = registry.load(args.env_name)

    wrapped_infer_env = wrapper.wrap_for_brax_training(
        infer_env,
        episode_length=ppo_params["episode_length"],
        action_repeat=ppo_params.get("action_repeat", 1),
    )

    rng = jax.random.split(
        jax.random.PRNGKey(ROLLOUT_SEED),
        NUM_VIDEOS,
    )

    reset_states = jax.jit(
        wrapped_infer_env.reset
    )(rng)

    empty_data = reset_states.data.__class__(
        **{
            k: None
            for k in reset_states.data.__annotations__
        }
    )

    empty_traj = reset_states.__class__(
        **{
            k: None
            for k in reset_states.__annotations__
        }
    )

    empty_traj = empty_traj.replace(
        data=empty_data
    )


    def rollout_step(carry, _):

        state, rng = carry

        rng, act_key = jax.random.split(rng)

        act_keys = jax.random.split(
            act_key,
            NUM_VIDEOS,
        )

        act = jax.vmap(
            jit_inference_fn
        )(
            state.obs,
            act_keys,
        )[0]

        state = wrapped_infer_env.step(
            state,
            act,
        )

        reward_components = {
            "orientation": state.metrics["reward/orientation"],
            "position": state.metrics["reward/position"],
            "termination": state.metrics["reward/termination"],
            "hand_pose": state.metrics["reward/hand_pose"],
            "action_rate": state.metrics["reward/action_rate"],
            "joint_vel": state.metrics["reward/joint_vel"],
            "energy": state.metrics["reward/energy"],
            "success": state.metrics["reward/success"],
            "total": state.reward,
        }

        traj_data = empty_traj.tree_replace({
            "data.qpos": state.data.qpos,
            "data.qvel": state.data.qvel,
            "data.time": state.data.time,
            "data.ctrl": state.data.ctrl,
            "data.mocap_pos": state.data.mocap_pos,
            "data.mocap_quat": state.data.mocap_quat,
            "data.xfrc_applied": state.data.xfrc_applied,
        })

        return (state, rng), (traj_data, reward_components)


    @jax.jit
    def do_rollout(state, rng):

        _, (traj, reward_history) = jax.lax.scan(
            rollout_step,
            (state, rng),
            None,
            length=ppo_params["episode_length"],
        )

        return traj, reward_history


    traj_stacked, reward_history = do_rollout(
        reset_states,
        jax.random.PRNGKey(
            ROLLOUT_SEED + 1
        ),
    )

    episode_reward_components = {
        key: jnp.sum(values, axis=0)
        for key, values in reward_history.items()
    }

    best_episode = int(
        np.asarray(
            jax.device_get(
                jnp.argmax(episode_reward_components["total"])
            )
        )
    )

    print(f"Best episode: {best_episode}\n")

    print("Reward breakdown of best episode:")
    print("---------------------------------")

    for key, values in episode_reward_components.items():
        value = float(
            np.asarray(
                jax.device_get(values[best_episode])
            )
        )
        print(f"{key:15s}: {value:10.2f}")
        
    traj_stacked = jax.tree.map(
        lambda x: jnp.moveaxis(x, 0, 1),
        traj_stacked,
    )

    trajectory = jax.tree.map(
        lambda x: x[best_episode],
        traj_stacked,
    )

    rollout = [
        jax.tree.map(
            lambda x, j=j: x[j],
            trajectory,
        )
        for j in range(
            ppo_params["episode_length"]
        )
    ]

    trajectories = [rollout]

    np.save(
        output_dir / "trajectories.npy",
        np.asarray(jax.device_get(trajectories), dtype=object),
        allow_pickle=True,
    )
    elapsed = datetime.now() - start
    print(f"Training finished in {elapsed}")

    with open(output_dir / "run_config.json", "w") as f:
        json.dump(
            {
                "env_name": args.env_name,
                "policy_type": args.policy_type,
                "num_timesteps": args.num_timesteps,
                "num_envs": args.num_envs,
                "num_evals": args.num_evals,
                "learning_rate": args.learning_rate,
                "seed": args.seed,
                "elapsed": str(elapsed),
            },
            f,
            indent=2,
        )

    print("Results:", output_dir)


if __name__ == "__main__":
    main()
