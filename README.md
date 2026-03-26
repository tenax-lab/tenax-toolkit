# Tenax Toolkit

A collection of Claude Code skills for [Tenax](https://github.com/tenax-lab/tenax), a JAX-based tensor network library for quantum many-body physics.

## Skills

### Getting Started
- **tenax-getting-started** — Install Tenax, configure JAX backends (CPU/CUDA/TPU/Metal), run your first calculation

### Core Concepts
- **tenax-tensor-ops** — Tensor creation, label-based contraction, SVD, QR, eigendecomposition
- **tenax-symmetry** — U(1)/Z_n symmetries, SymmetricTensor, block-sparse operations, charge conservation
- **tenax-blueprint** — Design tensor network contractions with NetworkBlueprint and TensorNetwork
- **tenax-autompo** — Build Hamiltonians from natural-language model descriptions via AutoMPO

### Algorithms
- **tenax-dmrg-workflow** — Finite DMRG, iDMRG, and 2D cylinder DMRG ground-state calculations
- **tenax-ipeps-workflow** — iPEPS for 2D quantum systems: simple update, AD optimization, excitation spectra
- **tenax-trg-workflow** — TRG/HOTRG for 2D classical statistical mechanics
- **tenax-observables** — Compute expectation values, correlations, entanglement entropy from MPS/iPEPS

### Tools
- **tenax-debugger** — Diagnose shape mismatches, JAX tracing issues, gradient problems, convergence failures
- **tenax-benchmark** — Performance benchmarks across CPU/CUDA/TPU/Metal backends
- **tenax-ed-comparator** — Compare exact diagonalization with DMRG to validate results
- **tenax-homework** — Generate scaffolded homework problems for graduate courses

### Migration Guides
- **tenax-migration-tenpy** — Migrate from TeNPy
- **tenax-migration-itensor** — Migrate from ITensor (Julia/C++)
- **tenax-migration-quimb** — Migrate from quimb
- **tenax-migration-cytnx** — Migrate from Cytnx

## Usage

Skills are invoked automatically by Claude Code when your question matches a skill's trigger. You can also invoke them explicitly:

```
/tenax-dmrg-workflow
/tenax-ipeps-workflow
/tenax-symmetry
```

## Requirements

- [Tenax](https://github.com/tenax-lab/tenax) installed in the project
- [Claude Code](https://claude.ai/code) CLI
