<!-- ## **TeaCache4HunyuanVideo** -->
# TeaCache4HunyuanVideo

[TeaCache](https://github.com/LiewFeng/TeaCache) can speedup [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) 2x without much visual quality degradation, in a training-free manner.

## 📈 Inference Latency Comparisons on a Single A800 GPU


|      Resolution       |        HunyuanVideo       |    TeaCache (0.1)    |     TeaCache (0.15)    |
|:---------------------:|:-------------------------:|:--------------------:|:----------------------:|
|         540p          |        ~18 min            |     ~11 min          |       ~8 min           |
|         720p          |        ~50 min            |     ~30 min          |       ~23 min          | 


## Usage

Follow [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) to clone the repo and finish the installation, then copy 'teacache_sample_video.py' in this repo to the HunyuanVideo repo. You can modify the thresh in line 220 to obtain your desired trade-off between latency and visul quality.

For single-gpu inference, you can use the following command:

```bash
cd HunyuanVideo

python3 teacache_sample_video.py \
    --video-size 720 1280 \
    --video-length 129 \
    --infer-steps 50 \
    --prompt "A cat walks on the grass, realistic style." \
    --flow-reverse \
    --use-cpu-offload \
    --save-path ./teacache_results
```

To generate a video with 8 GPUs, you can use the following command:

```bash
cd HunyuanVideo

torchrun --nproc_per_node=8 teacache_sample_video.py \
    --video-size 1280 720 \
    --video-length 129 \
    --infer-steps 50 \
    --prompt "A cat walks on the grass, realistic style." \
    --flow-reverse \
    --seed 42 \
    --ulysses-degree 8 \
    --ring-degree 1 \
    --save-path ./teacache_results
```



## Acknowledgements

We would like to thank the contributors to the [HunyuanVideo](https://github.com/Tencent/HunyuanVideo).