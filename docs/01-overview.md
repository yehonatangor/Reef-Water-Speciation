# Overview

Recovering seawater carbonate speciation from an acid titration, at the accuracy the physics permits and at a speed an embedded controller can sustain.

## The measurement

Reef aquarium stability depends on maintaining seawater alkalinity within narrow limits to sustain coral calcification. Coral skeletons are calcium carbonate, and how readily they form depends on the concentrations of a few dissolved ions. Those concentrations cannot be measured directly by anything inexpensive. pH can.

A titration therefore adds strong acid to a weighed water sample in small increments and records the pH after each. The pH falls unevenly: it resists wherever a dissolved buffer absorbs protons, then drops sharply once that buffer is spent. The shape of the resulting curve encodes both the buffer capacity and chemical composition of the sample.

## The problem and what we built

Composition to curve is settled physical chemistry, and this package implements it exactly using thermodynamic constants sourced from primary literature and benchmarked against reference implementations.

Curve to composition has no closed form. The standard approach is non-linear numerical optimization: propose a composition, simulate its curve, compare, adjust, repeat. While analytical laboratories use high-precision automated burettes and certified electrodes, and commercial monitors rely on expensive proprietary hardware, hobbyists are typically restricted to consumer-grade pH probes, basic microcontrollers, and low-cost peristaltic pumps. Running numerical solvers on low-cost microcontrollers is computationally prohibitive, ruling out the hardware a hobbyist would actually own.

This project investigates whether computational modeling can compensate for physical hardware constraints. Inexpensive sensors introduce structured, deterministic errors rather than purely random noise. Non-ideal behaviors such as reference potential drift, sub-Nernstian electrode slopes, peristaltic delivery variations, sensor lag, and ADC quantization each exhibit distinct mathematical profiles. By fitting these systematic distortions alongside governing chemical equilibrium equations, the system extracts true alkalinity directly from degraded measurement signals.

A companion simulation engine generates synthetic training datasets that combine ground-truth chemical states with realistic hardware failure modes and sensor noise.

## The approach

Three components, in dependency order:

1. **A verified forward model.** 

  Every equilibrium constant is traced to its primary source, checked against published check values, and validated against an independent implementation of the same equations.

2. **An information-theoretic floor.** 

  A Cramér–Rao bound states the accuracy no estimator can exceed, given the noise profile. This reframes the question from whether a method is good to how close it sits to the theoretical floor.

3. **A learned inverse.** 

  A lightweight convolutional encoder predicts physical parameters (chemical concentrations and sensor calibration offsets), followed by a non-trainable, differentiable chemical equilibrium layer as a decoder that reconstructs the theoretical titration curve. A single forward pass replaces numerical optimization.

Thermodynamic validity is enforced directly by the model architecture:
- The network outputs a compact set of physical parameters representing chemical concentrations and sensor calibration offsets.
- A non-trainable, differentiable chemical equilibrium layer reconstructs the theoretical titration curve from these parameters.
- The loss function evaluates the residual between the reconstructed curve and the raw input signal.

Because predictions pass through an explicit forward chemistry model, the network cannot output thermodynamically impossible states.

## What the floor implies

The least-squares fit already sits at the Cramér–Rao bound. A learned model therefore cannot be meaningfully more accurate on ordinary seawater, and one that appears to be has acquired information it will not hold at inference time.

The goal is consequently speed at matching accuracy, with one key exception: water containing dissolved organic matter (DOM). Classical titration models assume buffering is driven solely by carbonate and borate, conflating organic acid contributions with carbonate alkalinity. By explicitly parameterizing organic buffering in the forward model, the system deconvolves organic contributions to isolate true bicarbonate alkalinity.

## Scope

Everything here is simulated. No physical instrument exists, and the noise model is built from component specifications and physical reasoning rather than from a calibration campaign against real hardware. Ground truth is exact because the data is synthetic, which is what makes the accuracy claims checkable.

The gap that remains is the obvious next step: build the instrument, run certified reference material through it, and establish which of these assumptions survive contact with a real electrode.

## Conventions

| Parameter | Convention / Unit |
|---|---|
| pH scale | total hydrogen ion |
| Concentration | $\mathrm{mol\,kg\text{-}soln^{-1}}$ |
| Temperature | $^\circ\mathrm{C}$ in interfaces, converted internally |
| Pressure | $1\ \mathrm{atm}$ |
| Reported quantities | $\mu\mathrm{mol\,kg^{-1}}$ |

Certified reference material reproducibility for a good alkalinity titration is $\pm 2\ \mu\mathrm{mol\,kg^{-1}}$. Every accuracy figure in this documentation should be read against that.