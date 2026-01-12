#!/usr/bin/env python
import argparse, pathlib, csv, sys, json, numpy as np, torch, jax, orbax.checkpoint as ocp
from jax.sharding import SingleDeviceSharding
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.common.policies.pi0.conversion_scripts.convert_pi0_to_hf_lerobot import (
    slice_initial_orbax_checkpoint,
    get_paligemma_config,
    get_gemma_config,
    slice_paligemma_state_dict,
    slice_gemma_state_dict,
    update_keys_with_prefix,
)
PRECISIONS = {"bfloat16": torch.bfloat16, "float32": torch.float32, "float16": torch.float16}

def flatten_np(tree, parent=""):
    out = {}
    for k, v in tree.items():
        nk = f"{parent}/{k}" if parent else k
        if isinstance(v, dict):
            out.update(flatten_np(v, nk))
        else:
            out[nk] = np.asarray(v)
    return out


def load_jax_flat(ckpt_dir):
    """returns flat dict of numpy arrays using the helper already in your converter"""
    full = slice_initial_orbax_checkpoint(ckpt_dir)  # returns {'paligemma_params': …, 'projection_params': …}
    paligemma_cfg = get_paligemma_config("float32")
    paligemma, gemma_raw = slice_paligemma_state_dict(full["paligemma_params"], paligemma_cfg)
    gemma_cfg = get_gemma_config("float32")
    gemma = slice_gemma_state_dict(gemma_raw, gemma_cfg)
    proj = {}
    for k, d in full["projection_params"].items():
        proj[f"{k}.kernel"] = np.asarray(d["kernel"]["value"] if isinstance(d["kernel"], dict) else d["kernel"])
        proj[f"{k}.bias"]   = np.asarray(d["bias"]["value"]   if isinstance(d["bias"],   dict) else d["bias"])
    # add prefixes so keys match the torch state_dict
    paligemma = update_keys_with_prefix(paligemma, "model.paligemma_with_expert.")
    gemma     = update_keys_with_prefix(gemma,     "model.paligemma_with_expert.")
    proj      = update_keys_with_prefix(proj,      "model.")

    proj_renamed = {}
    for old_key, arr in proj.items():
        if old_key.endswith(".kernel"):
            proj_renamed[old_key.replace(".kernel", ".weight")] = arr.T
        else:
            proj_renamed[old_key] = arr.T
    proj = proj_renamed
    return {**paligemma, **gemma, **proj}

def load_torch_flat(torch_dir):
    model = PI0Policy.from_pretrained(torch_dir)
    return {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}

import gc
def compare(jax_d, torch_d, atol, rtol, csv_path="diff_report.csv"):
    keys_jax, keys_torch = set(jax_d), set(torch_d)
    common, only_jax, only_torch = sorted(keys_jax & keys_torch), sorted(keys_jax-keys_torch), sorted(keys_torch-keys_jax)
    print(f"Common tensors: {len(common)}  |  Only-in-JAX: {len(only_jax)}  |  Only-in-Torch: {len(only_torch)}")
    if only_jax:   print("⚠️  Missing in Torch:", only_jax[:10], "...")
    if only_torch: print("⚠️  Extra   in Torch:", only_torch[:10], "...")

    rows = [("key","shape","dtype_jax","dtype_torch","mean_diff","max_diff","equal")]
    for k in common:
        a, b = jax_d[k], torch_d[k]
        if a.shape != b.shape:
            rows.append((k, f"{a.shape}->{b.shape}", a.dtype, b.dtype, "", "", "shape_mismatch"))
            continue
        a_np = np.asarray(a, dtype=np.float32)            # already ndarray
        b_np = (b.detach().cpu().float().numpy()          # tensor → ndarray
                if isinstance(b, torch.Tensor) else
                np.asarray(b, dtype=np.float32))
        diff = np.abs(a_np - b_np)

        equal = np.array_equal(a, b)
        rows.append((k, a.shape, str(a.dtype), str(b.dtype), f"{diff.mean():.3e}", f"{diff.max():.3e}", str(equal)))
        if not equal:
            print(f"❌ {k:80}  mean={diff.mean():.2e}  max={diff.max():.2e}")
        del a_np, b_np, diff, a, b
        gc.collect()
    # save CSV
    with open(csv_path,"w",newline="") as f: csv.writer(f).writerows(rows)
    print(f"📄 Detailed report saved to {csv_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--jax_ckpt", default = "/mnt/lustre-grete/usr/u12045/vla/duci/openpi/checkpoints/pi0_calvin_50%_joint/pi0_calvin_50%_joint/30000/params", required=False)
    p.add_argument("--torch_dir", default= "/mnt/lustre-grete/usr/u12045/vla/duci/VLA-Humanoid/calvin_jaxcp_conversion_to_torch" , required=False)
    # p.add_argument("--jax_ckpt", default = "/mnt/lustre-grete/usr/u12045/vla/hf_cache/openpi/openpi-assets/checkpoints/pi0_base/params", required=False)
    # p.add_argument("--torch_dir", default= "/mnt/lustre-grete/usr/u12045/vla/pi0_torch_newcp" , required=False)
    p.add_argument("--precision", choices=list(PRECISIONS), default="float32")
    p.add_argument("--atol", type=float, default=1e-5)
    p.add_argument("--rtol", type=float, default=1e-3)
    args = p.parse_args()

    print("🔄 Restoring checkpoints …")
    jax_flat   = load_jax_flat(args.jax_ckpt)
    torch_flat = load_torch_flat(args.torch_dir)
    compare(jax_flat, torch_flat, args.atol, args.rtol)