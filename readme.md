# A Fast Multimodal Data Loading Pipeline (FastDLP)

Training long video generation model or large-scale robot learning models presents significant data loading challenges. Whether training in-context robot learning models that require processing thousands of frames per sequence, or video generation models that need to synthesize extended temporal horizons, the bottleneck often lies in efficiently loading and processing these massive multimodal datasets. Prior data loading approaches struggle to keep up with modern GPU training speeds when handling hours of synchronized video, proprioception, and action data.

## Overview
FastDLP is designed to maximize data loading throughput for multimodal data, with a particular focus on handling long video sequences efficiently. The pipeline achieves high-performance through:

- Optimized video frame loading using parallel jpg reading and decoding 
- Efficient batch collation with minimal memory copies

The high throughput enables training on large-scale robot datasets with hundreds of hours of video data while keeping GPU utilization near 100%. The pipeline is particularly optimized for sequences of 1000+ timesteps common in robotic manipulation tasks.

## Installation
```bash
pip install -e .
```
