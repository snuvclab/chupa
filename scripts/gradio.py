import os, sys
import argparse

import pickle as pkl
from pathlib import Path
from datetime import datetime
from omegaconf import OmegaConf
from functools import partial
import numpy as np
import gradio as gr

from ldm.models.diffusion.image_editor import ImageEditor
from img_utils import load_smpl_images, frontback_img, merge_frontback

# from normal_nds.new_renderer import SMPLX_Renderer
from normal_nds.reconstruct import NormalNDS #normal_nds
from normal_nds.nds.utils import write_mesh, load_smpl_info

from chupa import Chupa

def demo(config):
    model = Chupa(config)
    with gr.Blocks(analytics_enabled=False) as chupa_inference:
        gr.Markdown("<div align='center'> <h3> Chupa: Carving 3D Clothed Humans from Skinned Shape Priors using 2D Diffusion Probabilistic Models </span> </h3> \
                     <a style='font-size:18px;color: steelblue; display: inline-block;' href='https://huggingface.co/papers/2305.11870'> [Code] </a> \
                    <a style='font-size:18px;color: steelblue; display: inline-block;' href='https://arxiv.org/abs/2305.11870'> [arXiv] </a> \
                    <a style='font-size:18px;color: steelblue; display: inline-block;' href='https://arxiv.org/pdf/2305.11870.pdf'> [pdf] </a></div>")
        
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    input_image = gr.Image(shape=(512,512), source='upload', type='numpy', label='input image')

                with gr.Row():
                    with gr.Column():
                        seed = gr.Number(42, label='seed')
                        steps = gr.Slider(minimum=10, maximum=100, value=20, step=10, label="Inference steps")
                    with gr.Column():
                        generate_btn = gr.Button("Generate")

                with gr.Row():
                    with gr.Accordion("Additional settings", open=False):
                        with gr.Column():
                            cfg_scale = gr.Slider(minimum=1, maximum=10, value=2, step=0.5, label="CFG scale")

                with gr.Row():   
                    with gr.Accordion("Text input", open=False):
                        with gr.Column():
                            use_text = gr.Checkbox(label="use text input?", value=False, interactive=config.gradio.use_text)
                            prompt = gr.Textbox()
                            negative_prompt = gr.Textbox(value="nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name")
                            image_cfg_scale = gr.Slider(minimum=1, maximum=10, value=1.5, step=0.5, label="Image CFG scale")

                with gr.Row():
                    with gr.Accordion("Closeup resample", open=False):
                        with gr.Column():
                            use_resample = gr.Checkbox(label="use resample?", value=False, interactive=config.gradio.use_resample)
                            resample_T = gr.Slider(minimum=0, maximum=1, value=0.2, step=0.05,  label="resample_T", interactive=config.gradio.use_resample)
                            n_resample = gr.Slider(minimum=0, maximum=10, value=1, step=1, label="n_resample", interactive=config.gradio.use_resample)
                            use_closeup = gr.Checkbox(label="use closeup?", value=False, interactive=config.gradio.use_closeup)
                            resample_T_face = gr.Slider(minimum=0, maximum=1, value=0.2, step=0.05, label="resample_T_face", interactive=config.gradio.use_closeup)
                            n_resample_face = gr.Slider(minimum=0, maximum=10, value=1, step=1, label="n_resample_face", interactive=config.gradio.use_closeup)
                
            with gr.Column():
                output_model = gr.Model3D()


        generate_btn.click(
            fn=model.forward_gradio, 
            inputs=[input_image, seed, steps, cfg_scale, image_cfg_scale, use_text, prompt, negative_prompt,
                    use_resample, resample_T, n_resample, use_closeup, resample_T_face, n_resample_face],
            outputs=[output_model],
        )
    return chupa_inference

if __name__ == '__main__':
    config_file = sys.argv[1]
    config = OmegaConf.load(config_file)
    config_cli = OmegaConf.from_cli()
    args = OmegaConf.merge(config, config_cli)  # update config from command line

    chupa_inference_demo = demo(args)
    chupa_inference_demo.launch()