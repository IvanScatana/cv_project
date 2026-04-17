import streamlit as st
import torch
import segmentation_models_pytorch as smp
from PIL import Image
import numpy as np
import torchvision.transforms as T
from huggingface_hub import hf_hub_download
import os
import tempfile

st.set_page_config(page_title="Инференс модели", layout="wide")
st.title("🤖 Предсказание на основе обученной модели")

# Кэшируем загрузку модели, чтобы не перезагружать её при каждом взаимодействии
@st.cache_resource
def load_model():
    # Загружаем модель с Hugging Face Hub
    # Предполагается, что файл называется best_unet.pth и лежит в репозитории Scatana/cv_project
    # Если имя файла другое, измените filename
    try:
        # Загружаем файл модели с Hugging Face
        model_path = hf_hub_download(
            repo_id="Scatana/cv_project",
            filename="best_unet.pth",
            cache_dir="./model_cache"  # можно указать свою папку для кэша
        )
    except Exception as e:
        st.error(f"Не удалось загрузить модель с Hugging Face: {e}")
        # Если не получилось, можно попробовать загрузить из локального файла
        # Для этого нужно предварительно скачать модель и положить её в папку models/
        model_path = "models/best_unet.pth"
        if not os.path.exists(model_path):
            st.error("Модель не найдена локально. Пожалуйста, проверьте путь.")
            return None
    
    # Создаём модель
    model = smp.Unet(
        encoder_name="resnet34",  # Измените, если использовался другой энкодер
        encoder_weights=None,      # Не используем предобученные веса
        in_channels=3,
        classes=1,
        activation=None
    )
    
    # Загружаем веса
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(model_path, map_location=device)
    
    # Если чекпоинт содержит ключ 'model_state_dict', извлекаем его
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    
    model.load_state_dict(state_dict, strict=False)  # strict=False на случай неполного чекпоинта
    model.to(device)
    model.eval()
    return model, device

# Загружаем модель
model, device = load_model()
if model is None:
    st.stop()

# Трансформации для предсказания
def transform_image(image):
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # добавляем размерность батча

# Функция предсказания
def predict_mask(model, image_tensor, device):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        pred = torch.sigmoid(output).squeeze().cpu().numpy()
        mask = (pred > 0.5).astype(np.uint8) * 255
        return mask, pred

# Интерфейс загрузки изображений
st.header("📤 Загрузите изображения для предсказания")
uploaded_files = st.file_uploader(
    "Выберите изображения",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Читаем изображение
        image = Image.open(uploaded_file).convert('RGB')
        st.subheader(f"Обработка: {uploaded_file.name}")
        
        # Показываем оригинал
        col1, col2 = st.columns(2)
        col1.image(image, caption="Оригинал", use_container_width=True)
        
        # Преобразуем и предсказываем
        input_tensor = transform_image(image)
        mask, prob = predict_mask(model, input_tensor, device)
        
        # Визуализируем маску
        col2.image(mask, caption="Предсказанная маска", use_container_width=True, clamp=True)
        
        # Опционально: показываем наложение
        overlay = np.zeros((*image.size[::-1], 3), dtype=np.uint8)
        overlay[:, :, 1] = mask  # зелёный канал
        blended = Image.blend(image, Image.fromarray(overlay), alpha=0.4)
        st.image(blended, caption="Наложение маски", use_container_width=True)
        
        st.markdown("---")
else:
    st.info("Загрузите одно или несколько изображений, чтобы увидеть результат работы модели.")