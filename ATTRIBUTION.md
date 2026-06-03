# Attribution

This repository is the public code release for the master's thesis
**"MCUFlowNet — Hardware-Aware Neural Architecture Search for Optical Flow on the
Grove Vision AI V2 (Cortex-M55 + Ethos-U55 microNPU)."**

## What is ours

The neural architecture search framework under [`EdgeFlowNAS/`](EdgeFlowNAS/) — the
three-level hardware–software co-design (macro-backbone redesign, search-space &
supernet, and the hybrid NSGA-II + LLM-agent hardware-in-the-loop search) — together
with the hardware-profiling and thesis-figure tooling under
[`EdgeFlowNet/sramTest/`](EdgeFlowNet/sramTest/), is our own work.

## What is the baseline (EdgeFlowNet)

[`EdgeFlowNet/`](EdgeFlowNet/) bundles a **modified copy of EdgeFlowNet**, the optical-flow
baseline this thesis builds on and compares against:

> Sai Ramana Kiran Pinnama Raju, Rishabh Singh, Manoj Velmurugan, and Nitin J. Sanket.
> *EdgeFlowNet: 100FPS@1W Dense Optical Flow For Tiny Mobile Robots.*
> IEEE Robotics and Automation Letters, 2024. DOI: 10.1109/LRA.2024.3496336.

- Upstream: https://github.com/pearwpi/EdgeFlowNet (vendored at commit `15b30c5`)
- Copyright (c) 2024 Perception and Autonomous Robotics (PeAR) Group, Worcester
  Polytechnic Institute. Licensed under the MIT License.
- The original license is preserved verbatim at
  [`EdgeFlowNet/LICENSE`](EdgeFlowNet/LICENSE), and the upstream README is kept at
  [`EdgeFlowNet/README.md`](EdgeFlowNet/README.md).
- The exact changes we made to the baseline are listed in
  [`EdgeFlowNet/MODIFICATIONS.md`](EdgeFlowNet/MODIFICATIONS.md).

The upstream EdgeFlowNet data-generation assets (Blender scenes, dataset bundler
tooling) are **not** re-hosted here; obtain them from the upstream repository if needed.

## License

Our code is released under the MIT License (see [`LICENSE`](LICENSE)), which is
compatible with and preserves the EdgeFlowNet baseline's own MIT License.
