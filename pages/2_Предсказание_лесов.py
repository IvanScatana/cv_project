import streamlit as st
import torch
import segmentation_models_pytorch as smp
from PIL import Image
import numpy as np
import torchvision.transforms as T
from huggingface_hub import hf_hub_download
import os

st.set_page_config(page_title="Инференс модели", layout="wide")
st.title("🤖 Предсказание на основе обученной модели")

@st.cache_resource
def load_model():
    # Загружаем модель с Hugging Face (файл best.pth)
    try:
        model_path = hf_hub_download(
            repo_id="Scatana/cv_project",
            filename="best_model.pth",
            cache_dir="./model_cache"
        )
    except Exception as e:
        st.error(f"Не удалось загрузить модель с Hugging Face: {e}")
        # Альтернативный путь: локальная папка
        model_path = "models/best_model.pth"
        if not os.path.exists(model_path):
            st.error("Модель не найдена ни на Hugging Face, ни локально.")
            return None, None
    
    # Архитектура должна соответствовать лучшей модели (эксперимент 6: resnet50)
    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        state_dict = torch.load(model_path, map_location=device)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Ошибка загрузки весов: {e}")
        return None, None

model, device = load_model()
if model is None:
    st.stop()

def transform_image(image):
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def predict_mask(model, image_tensor, device):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        prob = torch.sigmoid(output).squeeze().cpu().numpy()
        mask = (prob > 0.5).astype(np.uint8) * 255
        return mask, prob

st.header("📤 Загрузите изображения для предсказания")
uploaded_files = st.file_uploader(
    "Выберите изображения",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert('RGB')
        st.subheader(f"Обработка: {uploaded_file.name}")
        col1, col2 = st.columns(2)
        col1.image(image, caption="Оригинал", use_container_width=True)
        
        input_tensor = transform_image(image)
        mask, prob = predict_mask(model, input_tensor, device)
        col2.image(mask, caption="Предсказанная маска", use_container_width=True, clamp=True)
        
        # Наложение маски (зелёным)
        overlay = np.zeros((*image.size[::-1], 3), dtype=np.uint8)
        overlay[:, :, 1] = mask
        blended = Image.blend(image, Image.fromarray(overlay), alpha=0.4)
        st.image(blended, caption="Наложение маски", use_container_width=True)
        st.markdown("---")
else:
    st.info("Загрузите одно или несколько изображений, чтобы увидеть результат работы модели.")