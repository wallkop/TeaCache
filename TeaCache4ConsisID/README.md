<!-- ## **TeaCache4ConsisID** -->
# TeaCache4ConsisID

[TeaCache](https://github.com/LiewFeng/TeaCache) can speedup [ConsisID](https://github.com/PKU-YuanGroup/ConsisID) 2x without much visual quality degradation, in a training-free manner.

## 📈 Inference Latency Comparisons on a Single H100 GPU

| ConsisID | TeaCache (0.1) | TeaCache (0.15) | TeaCache (0.20) |
| :------: | :------------: | :-------------: | :-------------: |
|  ~110 s  |     ~70 s      |      ~53 s      |      ~41 s      |


## Usage

Follow [ConsisID](https://github.com/PKU-YuanGroup/ConsisID) to clone the repo and finish the installation, then you can modify the `rel_l1_thresh` to obtain your desired trade-off between latency and visul quality, and change the `ckpts_path`, `prompt`, `image` to customize your identity-preserving video.

For single-gpu inference, you can use the following command:

```bash
cd TeaCache4ConsisID

python3 teacache_sample_video.py \
    --rel_l1_thresh 0.1 \
    --ckpts_path BestWishYsh/ConsisID-preview \
    --image "https://github.com/PKU-YuanGroup/ConsisID/blob/main/asserts/example_images/2.png?raw=true" \
    --prompt "The video captures a boy walking along a city street, filmed in black and white on a classic 35mm camera. His expression is thoughtful, his brow slightly furrowed as if he's lost in contemplation. The film grain adds a textured, timeless quality to the image, evoking a sense of nostalgia. Around him, the cityscape is filled with vintage buildings, cobblestone sidewalks, and softly blurred figures passing by, their outlines faint and indistinct. Streetlights cast a gentle glow, while shadows play across the boy\'s path, adding depth to the scene. The lighting highlights the boy\'s subtle smile, hinting at a fleeting moment of curiosity. The overall cinematic atmosphere, complete with classic film still aesthetics and dramatic contrasts, gives the scene an evocative and introspective feel." \
    --seed 42 \
    --num_infer_steps 50 \
    --output_path ./teacache_results
```

To generate a video with 8 GPUs, you can use the following [here](https://github.com/PKU-YuanGroup/ConsisID/tree/main/tools).

## Resources

Learn more about ConsisID with the following resources.
- A [video](https://www.youtube.com/watch?v=PhlgC-bI5SQ) demonstrating ConsisID's main features.
- The research paper, [Identity-Preserving Text-to-Video Generation by Frequency Decomposition](https://hf.co/papers/2411.17440) for more details.

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

We would like to thank the contributors to the [ConsisID](https://github.com/PKU-YuanGroup/ConsisID).