# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import torch
from diffusers import LCMScheduler, AutoPipelineForText2Image
from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        cache_dir = "model_cache"
        local_files_only = True  # set to True if the model is cached
        model_id = "segmind/Segmind-Vega"
        adapter_id = "segmind/Segmind-VegaRT"
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            use_safetensors=True,
            variant="fp16",
        )
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to("cuda")

        self.pipe.load_lora_weights(
            adapter_id,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.pipe.fuse_lora()

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default="(worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch)",
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=100, default=4
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        generator = torch.Generator(device="cuda").manual_seed(seed)

        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            generator=generator,
            guidance_scale=0,
        ).images[0]

        out_path = "/tmp/out.png"
        image.save(out_path)
        return Path(out_path)
