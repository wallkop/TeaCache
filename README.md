# Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model

<div class="is-size-5 publication-authors", align="center",>
            <span class="author-block">
              <a href="https://liewfeng.github.io" target="_blank">Feng Liu</a><sup>1</sup><sup>*</sup>,&nbsp;
            </span>
            <span class="author-block">
              <a href="https://scholar.google.com.hk/citations?user=ZO3OQ-8AAAAJ" target="_blank">Shiwei Zhang</a><sup>2</sup>,&nbsp;
            </span>
            <span class="author-block">
              <a href="https://jeffwang987.github.io" target="_blank">Xiaofeng Wang</a><sup>1,3</sup>,&nbsp;
            </span>
            <span class="author-block">
              <a href="https://weilllllls.github.io" target="_blank">Yujie Wei</a><sup>4</sup>,&nbsp;
            </span>
            <span class="author-block">
              <a href="http://haonanqiu.com" target="_blank">Haonan Qiu</a><sup>5</sup>
            </span>
            <br>
            <span class="author-block">
              <a href="https://callsys.github.io/zhaoyuzhong.github.io-main" target="_blank">Yuzhong Zhao</a><sup>1</sup>,&nbsp;
            </span>
            <span class="author-block">
              <a href="https://scholar.google.com.sg/citations?user=16RDSEUAAAAJ" target="_blank">Yingya Zhang</a><sup>2</sup>,&nbsp;
            </span>
            <span class="author-block">
              <a href="https://scholar.google.com/citations?user=tjEfgsEAAAAJ&hl=en&oi=ao" target="_blank">Qixiang Ye</a><sup>1</sup>,&nbsp;
            </span>
            <span class="author-block">
              <a href="https://scholar.google.com/citations?user=0IKavloAAAAJ&hl=en&oi=ao" target="_blank">Fang Wan</a><sup>1</sup><sup>†</sup>
            </span>
          </div>

<div class="is-size-5 publication-authors", align="center">
            <span class="author-block"><sup>1</sup>University of Chinese Academy of Sciences,&nbsp;</span>
            <span class="author-block"><sup>2</sup>Alibaba Group</span>
            <br>
            <span class="author-block"><sup>3</sup>Institute of Automation, Chinese Academy of Sciences</span>
            <br>
            <span class="author-block"><sup>4</sup>Fudan University,&nbsp;</span>
            <span class="author-block"><sup>5</sup>Nanyang Technological University</span>
          </div>


<div class="is-size-5 publication-authors", align="center">
            (* Work was done during internship at Alibaba Group. † Corresponding author.)
          </div>

<div class="is-size-5 publication-authors", align="center">
    <a href="https://arxiv.org/abs/2411.19108">Paper</a> | 
    <a href="https://liewfeng.github.io/TeaCache/">Project Page</a>
</div>

![visualization](./assets/tisser.png)

## Latest News 🔥
- [2024/12/26] 🔥 Support [ConsisID](https://github.com/PKU-YuanGroup/ConsisID).
- [2024/12/24] 🔥 Support [HunyuanVideo](https://github.com/Tencent/HunyuanVideo).
- [2024/12/19] 🔥 Support [CogVideoX](https://github.com/THUDM/CogVideo).
- [2024/12/06] 🎉 Release the [code](https://github.com/LiewFeng/TeaCache) of TeaCache. Support [Open-Sora](https://github.com/hpcaitech/Open-Sora), [Open-Sora-Plan](https://github.com/PKU-YuanGroup/Open-Sora-Plan) and [Latte](https://github.com/Vchitect/Latte).
- [2024/11/28] 🎉 Release the [paper](https://arxiv.org/abs/2411.19108) of TeaCache.

## Introduction
We introduce Timestep Embedding Aware Cache (TeaCache), a training-free caching approach that estimates and leverages the fluctuating differences among model outputs across timesteps. For more details and visual results, please visit our [project page](https://github.com/LiewFeng/TeaCache).

## TeaCache for HunyuanVideo
Please refer to [TeaCache4HunyuanVideo](./TeaCache4HunyuanVideo/README.md).

## TeaCache for ConsisID

Please refer to [TeaCache4ConsisID](./TeaCache4ConsisID/README.md).

## Installation

Prerequisites:

- Python >= 3.10
- PyTorch >= 1.13 (We recommend to use a >2.0 version)
- CUDA >= 11.6

We strongly recommend using Anaconda to create a new environment (Python >= 3.10) to run our examples:

```shell
conda create -n teacache python=3.10 -y
conda activate teacache
```

Install TeaCache:

```shell
git clone https://github.com/LiewFeng/TeaCache
cd TeaCache
pip install -e .
```


## Evaluation of TeaCache

We first generate videos according to VBench's prompts.

And then calculate Vbench, PSNR, LPIPS and SSIM based on the video generated.

1. Generate video
```
cd eval/teacache
python experiments/latte.py
python experiments/opensora.py
python experiments/open_sora_plan.py
python experiments/cogvideox.py
```

2. Calculate Vbench score
```
# vbench is calculated independently
# get scores for all metrics
python vbench/run_vbench.py --video_path aaa --save_path bbb
# calculate final score
python vbench/cal_vbench.py --score_dir bbb
```

3. Calculate other metrics
```
# these metrics are calculated compared with original model
# gt video is the video of original model
# generated video is our methods's results
python common_metrics/eval.py --gt_video_dir aa --generated_video_dir bb
```

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

## Acknowledgement

This repository is built based on [VideoSys](https://github.com/NUS-HPC-AI-Lab/VideoSys), [Open-Sora](https://github.com/hpcaitech/Open-Sora), [Open-Sora-Plan](https://github.com/PKU-YuanGroup/Open-Sora-Plan), [Latte](https://github.com/Vchitect/Latte), [CogVideoX](https://github.com/THUDM/CogVideo), [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) and [ConsisID](https://github.com/PKU-YuanGroup/ConsisID). Thanks for their contributions!
