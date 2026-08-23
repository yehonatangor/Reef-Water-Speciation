# Overview

## The problem and what we built

Reef aquarium stability depends on maintaining seawater alkalinity within narrow limits to sustain coral calcification. Precise measurement relies on potentiometric acid titration, where the resulting curve reflects both the buffer capacity and chemical composition of the sample. While analytical laboratories use high-precision automated burettes and certified electrodes, and commercial monitors rely on expensive proprietary hardware, hobbyists are typically restricted to consumer-grade pH probes, basic microcontrollers, and low-cost peristaltic pumps.

This project investigates whether computational modeling can compensate for these physical hardware constraints. Inexpensive sensors introduce structured, deterministic errors rather than purely random noise. Non-ideal behaviors such as reference potential drift, sub-Nernstian electrode slopes, peristaltic delivery variations, sensor lag, and ADC quantization each exhibit distinct mathematical profiles. By fitting these systematic distortions alongside governing chemical equilibrium equations, the system extracts true alkalinity directly from degraded measurement signals.

The system is built on a verified chemical equilibrium library that maps solution composition to expected titration curves, using thermodynamic constants sourced from primary literature and benchmarked against reference implementations. A companion simulation engine generates synthetic training datasets that combine ground-truth chemical states with realistic hardware failure modes and sensor noise.

While iterative non-linear optimization provides an accurate analytical baseline near theoretical precision limits, running numerical solvers on low-cost microcontrollers is computationally difficult. To resolve this, the framework uses a lightweight neural network operating in a single forward pass.

Thermodynamic validity is enforced directly by the model architecture:

- The network outputs a compact set of physical parameters representing chemical concentrations and sensor calibration offsets.

- A non-trainable, differentiable chemical equilibrium layer reconstructs the theoretical titration curve from these parameters.

- The loss function evaluates the residual between the reconstructed curve and the raw input signal.

Because predictions pass through an explicit forward chemistry model, the network cannot output thermodynamically impossible states. In addition, explicitly parameterizing dissolved organic matter allows the system to deconvolve organic acid buffering, resolving interferences that standard titration methods conflate with bicarbonate alkalinity.