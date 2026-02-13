# Project Context: PINN-Based Inverse Design for Ultrasound Haptics

**Objective:** Publishable research (Nature Comm. level) on optimizing "Asymmetric Spatiotemporal Focusing" to maximize shear force while minimizing shock wave wakes.

## 1. Core Philosophy & Physics

- **The Problem:** Conventional methods (AM/LM) create "Shear Shock Waves" (Mach cones) that blur tactile sensation (Reardon et al., 2023).
- **The Solution:** Inverse Design via PINN. Utilize **manipulation of viscoelastic shear waves**.
- **The Mechanism:** "Taming the Shock." Optimize a trajectory so that maximum shear force is produced while minimizing shock wave wakes. 最终在皮肤表面的目标点位置产生200Hz（帕西尼小体敏感频率）频率的脉冲高强度剪切，同时抑制其他位置的剪切强度。从而同时提高触觉强度和精度。
- **Key Physics:** Viscoelastic Wave Equation (Kelvin-Voigt model). Shear waves are slower (~2-10 m/s) and dispersive compared to pressure waves.

## 2. Hardware Constraints (Immutable)

- **Controller:** STM32 H750 + FPGA.
- **Array:** 60 transducers (40kHz), Hexagonal packing (side=5 units, center removed).
- **Update Cycle:** Fixed 200Hz loop (5ms duration).
- **Resolution:** Exactly **200 time steps** per cycle (dt = 25us).
- **Output:** The PINN must optimize a trajectory tensor of shape `[200, 2]` representing $(x, y)$ coordinates for the SINGLE focal point at each timestep. No multi-foci allowed due to power constraints.

## 3. Engineering Stack

- **Language:** Python 3.12.
- **Virtual Environment:** Miniconda (virtual environment name is UMH-PINN).
- **Core ML Framework:** PyTorch (Must use Autograd).
- **Physics Engine:** **Custom Differentiable FDTD Layer** (written in PyTorch) to simulate 2D Viscoelastic waves. *Note: Do not wrap external K-Wave binaries as they break the gradient chain.*
- **Optimization:** Adam optimizer.

## 4. Coding Standards

- **Vectorization:** Avoid `for` loops in physics simulation; use PyTorch tensor operations.
- **Modularity:** Separate Physics Layer, Model Architecture, and Loss Function.
- **Units:** SI Units (meters, seconds, Pascals) throughout.
- **Reproducibility:** Fix random seeds.
- **Comments:** Add comments only when very necessary.
