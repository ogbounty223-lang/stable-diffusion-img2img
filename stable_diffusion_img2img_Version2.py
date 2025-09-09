from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import torch

# تحميل النموذج (يحتاج اتصال بالإنترنت أول مرة)
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")  # إذا توفر كارت شاشة NVIDIA

# تحميل صورة الإدخال وتعديل حجمها
init_image = Image.open("input.jpg").convert("RGB")
init_image = init_image.resize((512, 512))

# وصف التحويل للحصول على صورة واقعية عالية الجودة
prompt = "make this image look realistic, high quality, photorealistic"
result = pipe(prompt=prompt, image=init_image, strength=0.8, guidance_scale=7.5)

# حفظ النتيجة
result.images[0].save("output.jpg")