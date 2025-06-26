import torch, gradio as gr
from diffusers import DiffusionPipeline
import tempfile, os

print("⏳ Downloading & loading LTX-Video …")
pipe = DiffusionPipeline.from_pretrained(
    "Lightricks/LTX-Video",          # تحميل مباشر من Hugging Face Hub
    torch_dtype=torch.float16,
)
try:
    pipe.enable_xformers_memory_efficient_attention()
except Exception as e:
    print(f"xformers disabled: {e}")

pipe.to("cuda")
print("✅ Model ready")

def img2vid(image, prompt):
    # احفظ الصورة في ملف مؤقت
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.save(tmp.name)
        img_path = tmp.name

    # استدعاء النموذج
    video_frames = pipe(
        prompt=prompt,
        conditioning_media_paths=[img_path],
        conditioning_start_frames=[0],
        num_frames=24,
        height=512,
        width=768,
    ).frames

    # حفظ الفيديو
    out_path = "/workspace/out.mp4"
    pipe.save_frames_as_video(video_frames, out_path, fps=24)

    # تنظيف الملف المؤقت
    os.remove(img_path)
    return out_path

demo = gr.Interface(
    fn=img2vid,
    inputs=[gr.Image(type="pil"), gr.Textbox(label="Prompt")],
    outputs=gr.Video(label="Generated video"),
    title="LTX-Video – Image ➜ Video",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
