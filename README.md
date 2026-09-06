# Start Right, Arrive Right: Asynchronous Execution via Initial Noise Selection

**PAINT** (Prefix-Anchored INiTial Noise) — training-free asynchronous execution of action-chunking robot policies via initial noise selection.

> ★ **Accepted** — CoRL 2026
>
> ★ **Spotlight** — Diff4RL @ RSS 2026
>
> arXiv:2606.19774

---

## Overview

Robot policies generate **action chunks** — short bursts of future motions. Because generation takes time, the robot keeps executing the current chunk while the next one is computed. By the time the new chunk arrives, the robot has already moved forward, so the new chunk's prefix must seamlessly continue the executed motion — or the robot lurches at every boundary.

Prior methods (e.g., Real-Time Chunking) fix this by **steering** the denoising trajectory toward the executed actions, which can pull the generated chunk off the policy's learned distribution.

**PAINT** takes a different approach: instead of correcting the trajectory mid-generation, we find the initial noise `x₀*` whose *unmodified* forward pass naturally produces a prefix-consistent chunk. No gradients, no retraining, no policy modification.

## Key idea

The prefix constraint requires:

```text
A_{t-1}[s + i] = A_t[i]   for i = 0 … d-1
```

PAINT satisfies this by:

1. Running a naive forward pass to get a free chunk
2. Building a target chunk with the executed prefix grafted in
3. Inverting the flow ODE (backward Euler) to recover the noise for that target
4. **Re-painting**: keep the inverted noise for the prefix positions, restore free noise for the tail
5. Running the unmodified policy from this chosen noise — the result is prefix-consistent by construction

## Results

Evaluated on **12 Kinetix simulation environments** and **6 real-world tasks** across three robot embodiments:

| Embodiment        | Tasks                                         |
| ----------------- | --------------------------------------------- |
| Single-arm        | Block Stacking, Toy in Drawer, Banana in Pot  |
| Bimanual (ALOHA)  | Towel Flinging, Shorts Folding                |
| Humanoid          | Part Placing                                  |

PAINT matches or exceeds RTC (gradient-based steering) on success rate and prefix consistency across GR00T-N1.5 and π₀ backbones — while never modifying the policy's velocity field.

## Citation

```bibtex
@inproceedings{ho2026start,
  title={Start Right, Arrive Right: Asynchronous Execution via Initial Noise Selection},
  author={Ho, Trong-Bao and Nguyen, Quang-Tan and Ha, Thien-Loc and Nguyen, Gia-Binh and Nguyen, Viet-Thanh and Dinh, Long and Vu, Minh N and Nguyen, Duy MH and Le, An Thai and Vien, Ngo Anh},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2026}
}
```
