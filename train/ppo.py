"""PPO for Pokemon Emerald exploration, written directly against GbaVecEnv.

    python -m train.ppo --rom emerald.gba --state train/states/emerald_start.state

One process holds both the emulator (Vulkan) and the policy (CUDA) on the same
GPU. Rollout observations are kept in host memory, since at thousands of
instances they would not fit beside the emulator in 8 GB; everything else lives
on the GPU.

GPU memory is the constraint to watch. When the emulator and PyTorch together
exceed the card, Windows pages PyTorch's allocations to system memory rather
than failing, and everything slows by two orders of magnitude: one forward pass
over 4096 instances reserved 3.3 GB and took 10 s instead of 33 ms. So the
policy runs in chunks while collecting, minibatches are sized in samples, and
both use bfloat16 autocast.
"""

import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from .emerald import ACTIONS, REWARD_TERMS, EmeraldExplore
from .grid import screens, write_png

MAP_BUCKETS = 1024


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rom", required=True)
    p.add_argument("--state", required=True, help="reset point every episode starts from")
    p.add_argument("--run", default="runs/emerald", help="directory for logs and checkpoints")
    p.add_argument("--resume", action="store_true", help="continue from the run's latest checkpoint")
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--frames-per-action", type=int, default=16, help="16 frames walks one tile")
    p.add_argument("--episode-steps", type=int, default=512)
    p.add_argument("--rollout", type=int, default=32, help="steps per environment per update")
    p.add_argument("--total-steps", type=float, default=2e8)
    p.add_argument("--frame-stack", type=int, default=3)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.998)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--minibatch-size", type=int, default=2048)
    p.add_argument("--infer-chunk", type=int, default=1024,
                   help="instances per policy forward pass while collecting")
    p.add_argument("--clip", type=float, default=0.1)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--checkpoint-every", type=int, default=10, help="updates between checkpoints")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--gpu-duty", type=float, default=0.5,
                   help="share of time the emulator may hold the GPU (0-1]; below 1 leaves room for the desktop")
    p.add_argument("--grid", type=int, default=36,
                   help="instances drawn to <run>/grid.png every step for train.watch; 0 to disable")
    return p.parse_args()


class Policy(nn.Module):
    """Nature-DQN CNN over stacked frames, plus an embedding of the current map.

    The 120x80 screen cannot resolve text and many maps look alike, so the map
    identifier from RAM is given to the network directly.
    """

    def __init__(self, stack: int, height: int, width: int, actions: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(stack, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat = self.cnn(torch.zeros(1, stack, height, width)).shape[1]
        self.map_embed = nn.Embedding(MAP_BUCKETS, 32)
        self.trunk = nn.Sequential(nn.Linear(flat + 32, 512), nn.ReLU())
        self.pi = nn.Linear(512, actions)
        self.v = nn.Linear(512, 1)
        nn.init.orthogonal_(self.pi.weight, 0.01)
        nn.init.orthogonal_(self.v.weight, 1.0)

    def forward(self, frames: torch.Tensor, maps: torch.Tensor):
        z = self.cnn(frames.float() / 255.0)
        z = self.trunk(torch.cat([z, self.map_embed(maps % MAP_BUCKETS)], dim=1))
        return self.pi(z), self.v(z).squeeze(1)


def act(policy: "Policy", frames: torch.Tensor, maps: torch.Tensor, chunk: int):
    """Samples actions for every instance, a chunk at a time to bound memory."""
    actions, logps, values = [], [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, frames.shape[0], chunk):
            logits, value = policy(frames[i:i + chunk], maps[i:i + chunk])
            dist = Categorical(logits=logits.float())
            a = dist.sample()
            actions.append(a)
            logps.append(dist.log_prob(a))
            values.append(value.float())
    return torch.cat(actions), torch.cat(logps), torch.cat(values)


class ReturnNormalizer:
    """Scales rewards by a running estimate of the discounted return's spread."""

    def __init__(self, n: int, gamma: float, device):
        self.gamma, self.ret = gamma, torch.zeros(n, device=device)
        self.mean, self.var, self.count = 0.0, 1.0, 1e-4

    def __call__(self, reward: torch.Tensor, done: bool) -> torch.Tensor:
        self.ret = self.ret * self.gamma + reward
        batch_mean, batch_var, n = self.ret.mean().item(), self.ret.var().item(), self.ret.numel()
        total = self.count + n
        delta = batch_mean - self.mean
        self.mean += delta * n / total
        self.var = (self.var * self.count + batch_var * n + delta ** 2 * self.count * n / total) / total
        self.count = total
        if done:
            self.ret.zero_()
        return reward / (self.var ** 0.5 + 1e-8)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.run, exist_ok=True)

    # Read by the emulator library when the environment is created.
    os.environ["GBA_ENV_GPU_DUTY"] = str(args.gpu_duty)
    env = EmeraldExplore(args.rom, args.state, args.num_envs, args.frames_per_action, args.episode_steps)
    n, k, T = env.n, args.frame_stack, args.rollout
    h, w = env.env.obs_height, env.env.obs_width

    policy = Policy(k, h, w, len(ACTIONS)).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
    global_step, update = 0, 0
    ckpt_path = os.path.join(args.run, "latest.pt")
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        policy.load_state_dict(ckpt["policy"])
        opt.load_state_dict(ckpt["opt"])
        global_step, update = ckpt["global_step"], ckpt["update"]
        print(f"resumed from {ckpt_path} at step {global_step}")

    # Rollout buffers. Frames stay on the host (T*N*k*H*W bytes).
    obs_buf = torch.zeros((T, n, k, h, w), dtype=torch.uint8).pin_memory()
    map_buf = torch.zeros((T, n), dtype=torch.long, device=device)
    act_buf = torch.zeros((T, n), dtype=torch.long, device=device)
    logp_buf = torch.zeros((T, n), device=device)
    rew_buf = torch.zeros((T, n), device=device)
    done_buf = torch.zeros(T, device=device)
    val_buf = torch.zeros((T, n), device=device)

    stack = torch.zeros((n, k, h, w), dtype=torch.uint8, device=device)
    stack[:, -1] = torch.from_numpy(env.observe()).to(device)
    maps = torch.from_numpy(env.map_ids()).to(device)
    normalize = ReturnNormalizer(n, args.gamma, device)

    # log.csv: one row per update, with the current episode's progress so far,
    # so learning is visible long before an episode ends.
    # episodes.csv: one row per finished episode, with its final numbers.
    stat_keys = (["episode_step", "return", "tiles", "maps", "max_maps", "party", "owned", "level_sum",
                  "badges", "story_flags"] + [f"r_{term}" for term in REWARD_TERMS])

    def open_csv(name, header):
        path = os.path.join(args.run, name)
        new = not os.path.exists(path)
        f = open(path, "a", newline="")
        writer = csv.writer(f)
        if new:
            writer.writerow(header)
        return f, writer

    log_file, log = open_csv("log.csv", ["update", "global_step", "sps"] + stat_keys +
                             ["policy_loss", "value_loss", "entropy", "approx_kl"])
    episodes_file, episodes = open_csv("episodes.csv", ["update", "global_step"] + stat_keys)

    batch = T * n
    while global_step < args.total_steps:
        t0 = time.time()
        # ---- collect ----------------------------------------------------
        for t in range(T):
            obs_buf[t].copy_(stack)  # blocking: stack is modified in place below
            map_buf[t] = maps
            action, logp, value = act(policy, stack, maps, args.infer_chunk)
            act_buf[t], logp_buf[t], val_buf[t] = action, logp, value

            frame, reward, done, info = env.step(action.cpu().numpy())
            rew_buf[t] = normalize(torch.from_numpy(reward).to(device), done)
            done_buf[t] = float(done)
            if done:
                stack.zero_()
                episodes.writerow([update + 1, global_step + (t + 1) * n] + [info[key] for key in stat_keys])
                episodes_file.flush()
            stack = torch.roll(stack, -1, dims=1)
            stack[:, -1] = torch.from_numpy(frame).to(device)
            maps = torch.from_numpy(env.map_ids()).to(device)
            if args.grid:
                write_png(os.path.join(args.run, "grid.png"), screens(env.env, args.grid))
        global_step += batch

        # ---- advantages -------------------------------------------------
        _, _, next_value = act(policy, stack, maps, args.infer_chunk)
        with torch.no_grad():
            adv = torch.zeros_like(rew_buf)
            gae = torch.zeros(n, device=device)
            for t in reversed(range(T)):
                nonterminal = 1.0 - done_buf[t]
                nv = next_value if t == T - 1 else val_buf[t + 1]
                delta = rew_buf[t] + args.gamma * nv * nonterminal - val_buf[t]
                gae = delta + args.gamma * args.gae_lambda * nonterminal * gae
                adv[t] = gae
            ret = adv + val_buf

        # ---- optimise ---------------------------------------------------
        b_obs = obs_buf.reshape(batch, k, h, w)
        b_map, b_act = map_buf.reshape(-1), act_buf.reshape(-1)
        b_logp, b_adv, b_ret = logp_buf.reshape(-1), adv.reshape(-1), ret.reshape(-1)
        mb = min(args.minibatch_size, batch)
        stats = []
        for _ in range(args.epochs):
            order = torch.randperm(batch)
            for start in range(0, batch, mb):
                idx = order[start:start + mb]
                gidx = idx.to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, value = policy(b_obs[idx].to(device, non_blocking=True), b_map[gidx])
                logits, value = logits.float(), value.float()
                dist = Categorical(logits=logits)
                logp = dist.log_prob(b_act[gidx])
                ratio = (logp - b_logp[gidx]).exp()
                a = b_adv[gidx]
                a = (a - a.mean()) / (a.std() + 1e-8)
                pg_loss = torch.max(-a * ratio, -a * ratio.clamp(1 - args.clip, 1 + args.clip)).mean()
                v_loss = 0.5 * (value - b_ret[gidx]).pow(2).mean()
                entropy = dist.entropy().mean()
                loss = pg_loss + args.vf_coef * v_loss - args.ent_coef * entropy
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                opt.step()
                with torch.no_grad():
                    kl = ((ratio - 1) - (logp - b_logp[gidx])).mean()
                stats.append([pg_loss.item(), v_loss.item(), entropy.item(), kl.item()])

        update += 1
        sps = batch / (time.time() - t0)
        s = np.mean(stats, axis=0)
        e = env.stats()
        log.writerow([update, global_step, round(sps)] + [e[key] for key in stat_keys] +
                     [f"{x:.5f}" for x in s])
        log_file.flush()
        print(f"update {update} step {global_step:,} {sps:,.0f} sps | "
              f"episode step {e['episode_step']}/{args.episode_steps}: return {e['return']:.2f} "
              f"tiles {e['tiles']:.1f} maps {e['maps']:.2f} (max {e['max_maps']}) party {e['party']:.2f} "
              f"| pg {s[0]:.4f} v {s[1]:.4f} ent {s[2]:.3f} kl {s[3]:.4f} "
              f"| torch {torch.cuda.max_memory_reserved() / 2**30:.1f} GiB",
              flush=True)

        if update % args.checkpoint_every == 0:
            torch.save({"policy": policy.state_dict(), "opt": opt.state_dict(),
                        "global_step": global_step, "update": update, "args": vars(args)}, ckpt_path)

    torch.save({"policy": policy.state_dict(), "opt": opt.state_dict(),
                "global_step": global_step, "update": update, "args": vars(args)}, ckpt_path)
    log_file.close()
    episodes_file.close()
    env.close()


if __name__ == "__main__":
    main()
