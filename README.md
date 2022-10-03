# PredictiveCodingNet

An implementation (in Julia) of Predictive Coding networks/graphs from [Learning on Arbitrary Graph Topologies via Predictive Coding](https://arxiv.org/abs/2201.13180). I also make a small extension of the proposed algoritm by implementing precision esstiamtion as described in [Predictive coding, precision and natural gradients
](https://arxiv.org/abs/2111.06942). The long term goal of this project is to find a principled way to use infromation from the precision (inverse of the covariance) matrix to grow a network that decomposes the signal/error into maximally independent components. This idea is roughly inspired by [The Cascade-Correlation Learning Architecture](https://proceedings.neurips.cc/paper/1989/file/69adc1e107f7f7d035d7baf04342e1ca-Paper.pdf)  
