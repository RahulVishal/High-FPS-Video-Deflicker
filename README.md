## **Efficient On-Device Flicker Removal in High FPS Videos with Synthetic Data Generation**

An implementation of the *Divide and Deflicker* framework: a data-driven, pleasingly parallel method for removing AC-light flicker from high-FPS (slow-motion) video by treating every pixel as an independent 1D time series.

> Paper: Haritas, Jasmita, Meeradevi, Bankapure, Vishal, Sadarangani. *Efficient On-Device Flicker Removal in High FPS Videos with Synthetic Data Generation.* Department of AIML, Ramaiah Institute of Technology, Bangalore.

---

## Motivation

Electric lights run on alternating current, so their intensity oscillates at mains frequency (50 Hz and up). The eye can't see this, but high-FPS capture on phones exposes it: at 480 FPS, roughly 1 in every 8 frames can be captured in complete darkness.

Existing deflickering approaches are a poor fit for mobile devices:

- **Flawed Atlas** (Lei et al., CVPR 2023) relies on iterative global optimization, which is prohibitively slow.
- **Fulari et al.** (WACV 2024) relies on dense optical flow and convex optimization.
- **SRGAN-PBAFT** (Shanmugaraja et al., 2025) targets single images and lacks temporal consistency.
- **BlazeBVD** (Qiu et al., 2024) uses a multi-stage pipeline of histogram analysis and spatiotemporal networks, which parallelizes less naturally than a direct time-series decomposition.

Paired flicker/no-flicker video data for AC-induced flicker is also almost nonexistent, and unsupervised approaches tend to be domain-specific.

## Key Idea

Instead of a single 4D spatiotemporal problem, treat the video as **millions of independent 1D time-series problems**:

1. **Divide**: split the video `(width, height, channels, time)` along the temporal axis and track each pixel's RGB values across frames, giving one waveform per pixel per channel.
2. **Deflicker**: pass short windows of each waveform through a small sequence model that outputs a flicker-free window, then reassemble the video.

Because model input and output are tiny (`1 x 1 x 1 x window_size`, typically a window of 10), the work reduces to highly optimized batched matrix multiplications, and it is easy to generate training data synthetically.

## Pipeline

```
Video
  -> Fourier analysis (determine window length)
  -> Segment into windows
  -> Normalize + mean-center each window
  -> Time-series model (LSTM or Transformer)
  -> Restore scale and bias
  -> Deflickered video
```

| Stage | Description |
|---|---|
| **Fourier analysis** | Estimates the flicker period to choose the window length. |
| **Segmentation** | Splits each pixel's time series into windows. |
| **Normalization / mean-centering** | Removes per-window scale and bias so the model sees a canonical input. |
| **Time-series model** | Removes the flicker from each window. |
| **Restore scale and bias** | Re-applies the stored statistics before reconstructing frames. |

### Models

- **LSTM**: learns sequential patterns in the waveform.
- **Transformer**: uses self-attention for efficient, parallelizable processing.

Both are deliberately compact for on-device use.

## Synthetic Data Generation

No real flicker/no-flicker video pairs are needed. Since windows are low-dimensional, the space of possible signals can be covered exhaustively:

1. **Generate clean target windows**: arrays covering the range of pixel intensity patterns, plus *desirable* oscillations and noise that must be **preserved** (e.g. a disco ball's intensity changes should not be removed).
2. **Generate model inputs**: add sinusoidal flicker with varied **frequencies, phases, and amplitudes** to each clean window.
3. **Train**: input = flickered window, target = clean window.

This lets the model learn to separate AC-light flicker from legitimate intensity variation, without depending on domain-specific video data.

## Results (as reported in the paper)

Total inference time on an identical high-FPS clip with significant flicker:

| Method | Time (s) |
|---|---|
| Flawed Atlas | 6171.0 |
| **Divide and Deflicker** | **134.8** |

That is a **~45x speedup**. Visual inspection confirmed the output was free of perceptible flicker.

Experiments in the paper were run on an Intel i9-10900X CPU, 32 GB RAM, and an NVIDIA RTX 3070 GPU.

> Numbers above are from the paper. Add your own benchmark results here once you've run the implementation.

## Limitations and Future Work

Per the paper, planned directions include:

- Richer synthetic data with more complex lighting, especially **multiple out-of-phase flicker sources**.
- More robust testing across **multiple devices**, with guidance on tuning performance parameters.
- Extending the approach to a general methodology for other **temporal artifacts** on mobile platforms.

## Citation

```bibtex
@inproceedings{haritas2025divide,
  title     = {Efficient On-Device Flicker Removal in High FPS Videos with Synthetic Data Generation},
  author    = {Haritas, Hrishikesh K and Jasmita, T. and Meeradevi and Bankapure, Darshan and Vishal, Rahul K and Sadarangani, Vineet H},
  booktitle = {<venue>},
  year      = {<year>}
}
```

## Authors

Hrishikesh K Haritas, T. Jasmita, Meeradevi, Darshan Bankapure, Rahul K Vishal, Vineet H Sadarangani, Department of AIML, Ramaiah Institute of Technology, Bangalore.

## References

1. C. Lei, X. Ren, Z. Zhang, Q. Chen. *Blind video deflickering by neural filtering with a flawed atlas.* CVPR 2023.
2. A. Fulari, S. Mulleti, A. Rajwade. *Unsupervised model-based learning for simultaneous video deflickering and deblotching.* WACV 2024.
3. T. Shanmugaraja et al. *A super resolution generative adversarial networks and partition-based adaptive filtering technique for detect and remove flickers in digital color images.* PLoS One 20(5), e0317758, 2025.
4. X. Qiu et al. *BlazeBVD: Make scale-time equalization great again for blind video deflickering.* arXiv:2403.06243, 2024.
