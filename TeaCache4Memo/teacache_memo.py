import torch
import numpy as np
from typing import Optional, Dict, Any
from memo.pipelines.video_pipeline import VideoPipeline
from memo.models.unet_3d import UNet3DConditionOutput
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers

logger = logging.get_logger(__name__)

def teacache_forward(
        self,
        sample: torch.FloatTensor,
        ref_features: dict,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        audio_embedding: torch.Tensor = None,
        audio_emotion: torch.Tensor = None,
        class_labels: torch.Tensor = None,
        mask_cond_fea: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        down_block_additional_residuals: Optional[tuple] = None,
        mid_block_additional_residual: torch.Tensor = None,
        uc_mask: torch.Tensor = None,
        return_dict: bool = True,
        is_new_audio: bool = True,
        update_past_memory: bool = False,
):
    """
    TeaCache-accelerated forward for UNet3DConditionModel.
    This function implements the TeaCache optimization strategy for the memo model.
    """
    # Pre-processing
    if attention_mask is not None:
        attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
        attention_mask = attention_mask.unsqueeze(1)

    # Center input sample if configured
    if self.config.center_input_sample:
        sample = 2 * sample - 1.0

    # Initial convolution
    sample = self.conv_in(sample)
    if mask_cond_fea is not None:
        sample = sample + mask_cond_fea

    # TeaCache optimization logic
    if getattr(self, "enable_teacache", False):
        modulated_inp = sample.clone()

        # Force compute on first or last step, otherwise check cache
        if self.cnt == 0 or self.cnt == self.num_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            if self.previous_modulated_input is not None:
                # Calculate relative L1 distance
                rel_l1 = (
                    (modulated_inp - self.previous_modulated_input).abs().mean()
                    / (self.previous_modulated_input.abs().mean() + 1e-8)
                ).item()
            else:
                rel_l1 = 999.0

            self.accumulated_rel_l1_distance += rel_l1

            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0

        # Update cache
        self.previous_modulated_input = modulated_inp
        self.cnt += 1
        if self.cnt == self.num_steps:
            self.cnt = 0

        # Use cached residual if available
        if not should_calc:
            if self.previous_residual is not None:
                sample = sample + self.previous_residual
            else:
                should_calc = True

    # Process down blocks
    down_block_res_samples = (sample,)
    for downsample_block in self.down_blocks:
        if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
            sample, res_samples = downsample_block(
                hidden_states=sample,
                temb=timestep,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask
            )
        else:
            sample, res_samples = downsample_block(
                hidden_states=sample,
                temb=timestep,
            )
        down_block_res_samples += res_samples

    # Add residuals if provided
    if down_block_additional_residuals is not None:
        new_down_block_res_samples = ()
        for down_block_res_sample, down_block_additional_residual in zip(
            down_block_res_samples, down_block_additional_residuals
        ):
            down_block_res_sample = down_block_res_sample + down_block_additional_residual
            new_down_block_res_samples += (down_block_res_sample,)
        down_block_res_samples = new_down_block_res_samples

    # Mid block processing
    sample = self.mid_block(
        sample,
        timestep,
        encoder_hidden_states=encoder_hidden_states,
        attention_mask=attention_mask
    )
    if mid_block_additional_residual is not None:
        sample = sample + mid_block_additional_residual

    # Process up blocks
    for upsample_block in self.up_blocks:
        res_samples = down_block_res_samples[-len(upsample_block.resnets):]
        down_block_res_samples = down_block_res_samples[:-len(upsample_block.resnets)]

        if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
            sample = upsample_block(
                hidden_states=sample,
                temb=timestep,
                res_hidden_states_tuple=res_samples,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask
            )
        else:
            sample = upsample_block(
                hidden_states=sample,
                temb=timestep,
                res_hidden_states_tuple=res_samples,
            )

    # Store original state for residual calculation if needed
    if getattr(self, "enable_teacache", False) and should_calc:
        original_sample = sample.clone()

    # Final processing
    sample = self.conv_out(sample)

    # Update residual cache if we did full computation
    if getattr(self, "enable_teacache", False) and should_calc:
        self.previous_residual = sample - original_sample

    if not return_dict:
        return (sample,)

    return UNet3DConditionOutput(sample=sample)

# Set up TeaCache for memo's UNet3DConditionModel
from memo.models.unet_3d import UNet3DConditionModel
UNet3DConditionModel.forward = teacache_forward

# Example usage (commented out - uncomment and modify as needed)
"""
num_inference_steps = 50
pipeline = VideoPipeline.from_pretrained("path/to/memo/model")

# Enable TeaCache
pipeline.diffusion_net.__class__.enable_teacache = True
pipeline.diffusion_net.__class__.cnt = 0
pipeline.diffusion_net.__class__.num_steps = num_inference_steps
pipeline.diffusion_net.__class__.rel_l1_thresh = 0.15  # Adjust for speed vs quality trade-off
pipeline.diffusion_net.__class__.accumulated_rel_l1_distance = 0
pipeline.diffusion_net.__class__.previous_modulated_input = None
pipeline.diffusion_net.__class__.previous_residual = None

# Generate video with TeaCache optimization
output = pipeline(
    prompt="Your prompt here",
    num_inference_steps=num_inference_steps,
)
"""
