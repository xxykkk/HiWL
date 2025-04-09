Deep image watermarking, which refers to enable imperceptible watermark embedding and reliable extraction in cover images, has shown to be effective for copyright protection of image assets. 
However, existing methods face limitations in simultaneously satisfying three essential criteria for generalizable watermarking: 1) invisibility (imperceptible hide of watermarks), 2) robustness (reliable watermark recovery under diverse conditions), and 3) broad applicability (low latency in watermarking process). To address these limitations, we propose a \textbf{H}ierarchical \textbf{W}atermark \textbf{L}earning (\textbf{HiWL}), a two-stage optimization that enable a watermarking model to simultaneously achieve three criteria. In the first stage, distribution alignment learning is designed to establish a common latent space with two constraints: 1) visual consistency between marked and non-marked images, and 2) information invariance across watermark latent representations. In this way, multi-modal inputs including watermark message (binary codes) and cover images (RGB pixels) can be well represented, ensuring the invisibility of watermarks and robustness in watermarking process thereby. The second stage employs generalized watermark representation learning to establish a disentanglement policy for separating watermarks from image content in RGB space. In particular, it strongly penalizes substantial fluctuations in separated RGB watermarks corresponding to identical messages. Consequently, HiWL effectively learns generalizable RGB-space watermark representations while maintaining broad applicability. 
Extensive experiments demonstrate the effectiveness of proposed method. In particular, it achieves 7.6\% higher accuracy in watermark extraction than existing methods, while maintaining extremely low latency (100K images processed in 8s). Our code will be publicly available.

## Updates
* [2025/4/9: init code base]

## Installation
This codebase has been developed with python version 3.9, PyTorch version 2.0.1, CUDA 11.7.

## Training
Training code will be available soon.
