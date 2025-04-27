# Parallelism Methods for LLM Training

## 1. Data Parallelism
- Description:
Entire model is copied on each device. Each processes a different batch of data.
- How it works:
Forward and backward passes are run independently on each device.
Gradients are averaged and model parameters are synchronized after each step.
- Pros: Simple, widely supported.
- Cons: Replicates full model and optimizer states on every device.

## 2. Model Parallelism
- Description:
Splits the model itself across multiple devices, often required when the model is too large for the memory of a single device.
- How it works:
Different layers or parts of layers are assigned to different devices.
Data is transferred between devices as it passes through the model.
- Pros: Allows very large models.
- Cons: Can be complex; communication overhead.

## 3. Pipeline Parallelism
- Description:
Divides the model into sequential stages, each allocated to a device. Batches are further divided into micro-batches that flow through the pipeline.
- How it works:
Devices run different chunks of the model, processing micro-batches in a staggered, assembly-line fashion.
- Pros: Keeps devices busier; enables larger models.
- Cons: Complex scheduling; pipeline bubbles can reduce efficiency.

## 4. Tensor Parallelism
- Description:
Splits internal computations (like large matrix multiplications) within layers across multiple devices.
- How it works:
Large tensors are partitioned.
Computations and results are distributed and aggregated across devices.
- Pros: Enables scaling individual layers across devices.
- Cons: High communication bandwidth required.

## 5. ZeRO (Zero Redundancy Optimizer)
- Description:
Not a standalone parallelism type, but a memory optimization technique that shards optimizer state, gradients, and model parameters across data-parallel processes.
- How it works:
Each device holds only a portion (“shard”) of optimizer states/gradients/parameters.
Reduces per-device memory usage drastically.
- Pros: Allows much larger model scaling; complements other parallelism forms.
- Cons: Adds complexity and requires specific frameworks (e.g., DeepSpeed).

## 6. Hybrid Parallelism
- Description:
Combination of two or more methods (e.g., Data + Tensor + ZeRO + Pipeline).
- How it works:
Designs utilize strengths of each approach to maximize efficiency and scalability.
- Pros: Scales models to extreme sizes (hundreds of billions of parameters or more).
- Cons: Most complex to design/implement.

| Parallelism Type      | How It Works                                                | Pros / Cons                             |
|------------------------|-------------------------------------------------------------|-----------------------------------------|
| Data Parallelism       | Full model copy per device; each processes different batch. <br> Gradients averaged & model synced after each step. | + Simple, widely used <br> - Redundant memory per GPU |
| Model Parallelism      | Model split across devices; layers or layer parts assigned. <br> Data moves across devices in forward/backward passes. | + Enables larger models <br> - Needs careful partitioning <br> - Inter-device communication overhead |
| Pipeline Parallelism   | Model divided into sequential stages by device. <br> Micro-batches pipelined through model stages. | + Better device utilization <br> - Pipeline bubbles inefficiency |
| Tensor Parallelism     | Computations (e.g., matrix mults) within layers are split. <br> Each device does a part and results are aggregated. | + Huge model/layer scaling <br> - Much communication |
| ZeRO (Zero Redundancy Optimizer) | Shards model params, gradients, optimizers across devices. <br> No full copy—each device holds only a fragment. | + Dramatic memory savings <br> + Combines with other methods <br> - Needs special frameworks (DeepSpeed) |
| Hybrid Parallelism     | Combinations (e.g., Data + Tensor + ZeRO + Pipeline). <br> Uses multiple techniques together. | + Ultra-large scale possible <br> - Most complex to orchestrate |
