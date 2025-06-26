import torch, gradio as gr
import numpy as np
import imageio.v3 as iio
from PIL import Image
from diffusers import LTXImageToVideoPipeline

# تحميل النموذج
print("⏳ Loading LTX-Video …")
pipe = (
    LTXImageToVideoPipeline.from_pretrained(
        "Lightricks/LTX-Video", torch_dtype=torch.float16
    )
    .to("cuda")
)
print("✅ Model ready")

def to_rgb_array(frame):
    """حوّل أي إطار إلى مصفوفة RGB بعمق 3 قنوات."""
    if isinstance(frame, Image.Image):
        return np.array(frame.convert("RGB"))
    arr = np.array(frame)
    if arr.ndim == 2:          # صورة رمادية
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:     # RGBA → RGB
        arr = arr[..., :3]
    return arr.astype(np.uint8)

def img2vid(image: Image.Image, prompt: str):
    image = image.convert("RGB").resize((768, 512))

    # توليد الفيديو
    out = pipe(image=image, prompt=prompt, num_frames=24)

    # توحيد الإطارات في قائمة مسطَّحة
    nested = out.frames
    flat_frames = []
    for item in nested:
        flat_frames.extend(item if isinstance(item, (list, tuple)) else [item])

    # تحويل كل إطار إلى مصفوفة RGB
    rgb_frames = [to_rgb_array(f) for f in flat_frames]

    # كتابة الفيديو
    out_path = "/workspace/out.mp4"
    iio.imwrite(out_path, rgb_frames, fps=24)
    return out_path

# واجهة Gradio
demo = gr.Interface(
    fn=img2vid,
    inputs=[gr.Image(type="pil"), gr.Textbox(label="Prompt")],
    outputs=gr.Video(label="Generated video"),
    title="LTX-Video – Image ➜ Video",
    description="حمّل صورة وأدخل Prompt لتحويلها إلى فيديو 24-fps.",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
