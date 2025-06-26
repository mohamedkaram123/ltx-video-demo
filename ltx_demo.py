import torch, gradio as gr
from diffusers import LTXImageToVideoPipeline

print("⏳ Downloading & loading LTX-Video …")
pipe = LTXImageToVideoPipeline.from_pretrained(
    "Lightricks/LTX-Video",
    torch_dtype=torch.float16,
)
try:
    pipe.enable_xformers_memory_efficient_attention()
except Exception as e:
    print(f"xformers disabled: {e}")

pipe.to("cuda")
print("✅ Model ready")

def img2vid(image, prompt):
    # تأكد من الحجم المناسب
    image = image.convert("RGB").resize((768, 512))
    video_frames = pipe(
        image=image,
        prompt=prompt,
        num_frames=24,      # طول الفيديو
        height=512,
        width=768,
    ).frames

    out_path = "/workspace/out.mp4"
    pipe.save_frames_as_video(video_frames, out_path, fps=24)
    return out_path

demo = gr.Interface(
    fn=img2vid,
    inputs=[gr.Image(type="pil"), gr.Textbox(label="Prompt")],
    outputs=gr.Video(label="Generated video"),
    title="LTX-Video – Image ➜ Video",
    description="حمّل صورة وأدخل Prompt لتحويلها إلى فيديو.",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
