# TeaCache4Memo

[TeaCache](https://github.com/LiewFeng/TeaCache) can speedup [Memo](https://github.com/memoavatar/memo) 2x without much visual quality degradation, in a training-free manner. The following video shows the results generated by TeaCache-Memo with various `rel_l1_thresh` values: 0 (original), 0.09 (1.5x speedup), 0.15 (2.0x speedup), 0.25 (2.5x speedup).

[Video comparison placeholder - add actual comparison video when available]

## 📈 Inference Latency Comparisons on a Single A800

|      Memo (Original)      |    TeaCache (0.09)    |    TeaCache (0.15)    |    TeaCache (0.25)    |
|:------------------------:|:--------------------:|:--------------------:|:--------------------:|
|        ~25 min          |      ~17 min        |      ~12 min        |      ~10 min        |

## Installation

1. Install the required dependencies:
```bash
pip install --upgrade diffusers[torch] transformers protobuf tokenizers sentencepiece imageio
```

2. Clone and install TeaCache:
```bash
git clone https://github.com/wallkop/TeaCache.git
cd TeaCache
pip install -e .
```

## Usage

You can modify the `rel_l1_thresh` in the script to obtain your desired trade-off between latency and visual quality. For single-gpu inference, you can use the following command:

```bash
python teacache_memo.py
```

For more detailed usage, you can import and set up TeaCache in your own code:

```python
from TeaCache4Memo.teacache_memo import teacache_forward
from memo.models.unet_3d import UNet3DConditionModel
from memo.pipelines.video_pipeline import VideoPipeline

# Replace the forward method
UNet3DConditionModel.forward = teacache_forward

# Initialize your pipeline
pipeline = VideoPipeline.from_pretrained("path/to/memo/model")

# Enable TeaCache with desired settings
pipeline.diffusion_net.__class__.enable_teacache = True
pipeline.diffusion_net.__class__.cnt = 0
pipeline.diffusion_net.__class__.num_steps = num_inference_steps  # e.g., 50
pipeline.diffusion_net.__class__.rel_l1_thresh = 0.15  # Adjust for speed/quality trade-off
pipeline.diffusion_net.__class__.accumulated_rel_l1_distance = 0
pipeline.diffusion_net.__class__.previous_modulated_input = None
pipeline.diffusion_net.__class__.previous_residual = None
```

2. Generate video with TeaCache optimization:

```python
output = pipeline(
    prompt="Your prompt here",
    num_inference_steps=50,  # Must match the num_steps set above
)
```

## Configuration

The key parameter for controlling the speed/quality trade-off is `rel_l1_thresh`:

- Lower values (e.g., 0.09) prioritize quality but with less speedup
- Higher values (e.g., 0.15) provide more speedup with potential minor quality impact
- Recommended range: 0.09-0.25

Example speedups:
- rel_l1_thresh = 0.09: ~1.5x speedup
- rel_l1_thresh = 0.15: ~2.0x speedup
- rel_l1_thresh = 0.25: ~2.5x speedup

## Citation
If you find TeaCache is useful in your research or applications, please consider giving us a star 🌟 and citing it by the following BibTeX entry.

```
@article{liu2024timestep,
  title={Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model},
  author={Liu, Feng and Zhang, Shiwei and Wang, Xiaofeng and Wei, Yujie and Qiu, Haonan and Zhao, Yuzhong and Zhang, Yingya and Ye, Qixiang and Wan, Fang},
  journal={arXiv preprint arXiv:2411.19108},
  year={2024}
}
```

## Acknowledgements

We would like to thank the contributors to the [Memo](https://github.com/memoavatar/memo) and [Diffusers](https://github.com/huggingface/diffusers).