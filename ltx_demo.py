import torch, gradio as gr
from diffusers import DiffusionPipeline

print("⏳ Loading LTX-Video …")
pipe = DiffusionPipeline.from_pretrained(
    "/workspace/LTX-Video/LTX-Video",
    torch_dtype=torch.float16,
    use_safetensors=True,
)
try:
    pipe.enable_xformers_memory_efficient_attention()
except Exception as e:
    print(f"xformers disabled: {e}")
pipe.to("cuda")
print("✅ Model ready")

def img2vid(image, prompt):
    image = image.convert("RGB").resize((768, 512))
    frames = pipe(image=image, prompt=prompt, num_frames=24).frames
    out_path = "/workspace/out.mp4"
    pipe.save_frames_as_video(frames, out_path, fps=24)
    return out_path

demo = gr.Interface(
    fn=img2vid,
    inputs=[gr.Image(type="pil"), gr.Textbox(label="Prompt")],
    outputs=gr.Video(label="Generated video"),
    title="LTX-Video – Image ➜ Video",
    description="حمّل صورة ثم أدخل Prompt لتحويلها إلى فيديو 24-fps.",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
