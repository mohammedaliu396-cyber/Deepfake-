# -* coding:UTF-8 -*
# !/usr/bin/env python
import numpy as np
import gradio as gr
from gradio import themes
import roop.globals
from roop.core import (
    start,
    decode_execution_providers,
    suggest_max_memory,
    suggest_execution_threads,
)
from roop.processors.frame.core import get_frame_processors_modules
from roop.utilities import normalize_output_path
import os
from PIL import Image

def swap_face(source_file, target_file, doFaceEnhancer, video_quality):
    source_path = "input.jpg"
    target_path = "target.mp4"

    # Save source image
    source_image = Image.fromarray(source_file)
    source_image.save(source_path)

    # Copy the target video to the working directory
    os.rename(target_file, target_path)

    print("source_path: ", source_path)
    print("target_path: ", target_path)
    roop.globals.video_quality = video_quality
    roop.globals.source_path = source_path
    roop.globals.target_path = target_path
    output_path = "output.mp4"
    roop.globals.output_path = normalize_output_path(
        roop.globals.source_path, roop.globals.target_path, output_path
    )
    if doFaceEnhancer:
        roop.globals.frame_processors = ["face_swapper", "face_enhancer"]
    else:
        roop.globals.frame_processors = ["face_swapper"]
    roop.globals.headless = True
    roop.globals.keep_fps = True
    roop.globals.keep_audio = True
    roop.globals.keep_frames = False
    roop.globals.many_faces = False
    roop.globals.video_encoder = "libx264"
    roop.globals.video_quality = 10
    roop.globals.max_memory = suggest_max_memory()
    roop.globals.execution_providers = decode_execution_providers(["cuda"])
    roop.globals.execution_threads = suggest_execution_threads()

    print(
        "start process",
        roop.globals.source_path,
        roop.globals.target_path,
        roop.globals.output_path,
    )

    for frame_processor in get_frame_processors_modules(
        roop.globals.frame_processors
    ):
        if not frame_processor.pre_check():
            return

    start()
    return output_path

with gr.Blocks(
    title="ROOP DeepFake Video",
    theme=themes.Soft(primary_hue="blue", secondary_hue="blue"),  # Apply blue theme
) as demo:
    gr.HTML(f'<div style="text-align: center;"><img src="https://www.dpu.ac.th/img/Logo_update_080720.png" width="200"></div>')  # Header image

    gr.Markdown(
        """
        ## Create DeepFake Video
        Upload a source image and a target video, select whether to enable face enhancer, many-face mode and adjust video quality. Then click 'Submit' to swap faces in the video.
        **Credits:**
        
        This study was conducted in the College of Creative Design and Entertainment Technology laboratory at Dhurakij Pundit University by Asst. Prof. Banyapon Poolsawas. The project builds upon the s0md3v/roop repository (https://github.com/s0md3v/roop) for face swapping experiments with video files in the Generative AI and Machine Learning course.
        """
    )

    with gr.Row():
        with gr.Column():
            source_image = gr.Image(label="Source Image", type="numpy")
            target_video = gr.Video(label="Target Video")
            face_enhancer = gr.Checkbox(label="Enable Face Enhancer")
            video_quality = gr.Slider(minimum=0, maximum=100, step=1, value=18, label="Output Video Quality (0 is Best, 100 is Worst)")
            submit = gr.Button("Submit")

        with gr.Column():
            output_video = gr.Video(label="Output Video")

    submit.click(swap_face, inputs=[source_image, target_video, face_enhancer, video_quality], outputs=output_video)


demo.launch()
