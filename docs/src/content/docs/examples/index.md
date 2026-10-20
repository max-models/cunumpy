---
title: "Worked examples"
description: "Complete programs that combine the pieces described in the guides."
---

Complete programs that combine the pieces described in the guides.

* [A portable diffusion solver](/cunumpy/examples/portable-script/): array-level code only,
  one script for CPU and GPU, with a command-line backend switch, timing and
  output at the boundaries.
* [Porting a particle-in-cell code](/cunumpy/examples/particle-pusher/): a small simulation
  whose kernels are ported to CUDA one at a time with a `KernelCatalog`,
  verified with parity tests, and checked for transfers.

For an MPI program with one rank per GPU, see the start-up sequence and halo
exchange in [Multi-GPU programs with MPI](/cunumpy/guides/mpi/).
