import torch, gradio as gr
import imageio.v3 as iio
import numpy as np
from diffusers import LTXImageToVideoPipeline

print("⏳ Downloading & loading LTX-Video …")
pipe = LTXImageToVideoPipeline.from_pretrained(
    "Lightricks/LTX-Video",
    torch_dtype=torch.float16,
)
pipe.to("cuda")
print("✅ Model ready")

def img2vid(image, prompt):
    # إعداد الصورة
    image = image.convert("RGB").resize((768, 512))

    # توليد الإطارات
    frames = pipe(image=image, prompt=prompt, num_frames=24).frames  # قائمة PIL

    # تحويل الإطارات إلى RGB وكتابتها في فيديو
    out_path = "/workspace/out.mp4"
    rgb_frames = [np.array(f.convert("RGB")) for f in frames]
    iio.imwrite(out_path, rgb_frames, fps=24)

    return out_path

demo = gr.Interface(
    fn=img2vid,
    inputs=[gr.Image(type="pil"), gr.Textbox(label="Prompt")],
    outputs=gr.Video(label="Generated video"),
    title="LTX-Video – Image ➜ Video",
    description="حمّل صورة وأدخل Prompt لتحويلها إلى فيديو 24-fps.",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
