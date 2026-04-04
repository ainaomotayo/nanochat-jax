# LinkedIn Post: NanoChat-JAX Scaling Laws

## Post

I measured neural scaling laws across 7 model sizes (886K to 6.03B parameters) and two frameworks -- and the power law held: L = 3.29 x N^(-0.027) on TinyShakespeare.

The setup: I ported Karpathy's NanoChat to JAX/Flax NNX -- a faithful, production-grade implementation with 180 tests -- then ran identical scaling law experiments on both PyTorch and JAX.

The verdict on JAX vs PyTorch for LLM training:

JAX is 1.2-2.5x faster on TPU. On a single GPU at small scale, the two are comparable. The real advantage isn't raw speed -- it's that JAX's functional paradigm forces you to write code that scales cleanly to multi-device setups from day one. If your training pipeline targets TPUs, JAX pays for itself immediately.

The scaling exponent (-0.027) is shallow compared to Chinchilla (-0.076), which is expected: TinyShakespeare is 1MB of text, not 300B tokens. Small data saturates fast. The power law still emerges cleanly, which is the interesting part.

Supported by Google TPU Research Cloud and the AI GDE TPU Sprint 2026.

Code: https://github.com/ainaomotayo/nanochat-jax

#MachineLearning #JAX #ScalingLaws #LLM #DeepLearning #TPU #MLEngineering #NLP
