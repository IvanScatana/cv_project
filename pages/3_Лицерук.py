import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import requests
from io import BytesIO
import re

st.set_page_config(page_title="Face Emoji Overlay", layout="wide")
st.title("Наложение эмодзи на лица")

# -------------------------------------------------------------------
# Получение PNG эмодзи из текста через CDN
# -------------------------------------------------------------------
def emoji_to_image_url(emoji_char, size=160):
    """Возвращает URL картинки эмодзи с заданным размером."""
    # Кодируем эмодзи в percent-encoding
    try:
        # Для современных эмодзи (например, 🤦‍♂️) нужен правильный URL encode
        # Используем библиотеку requests.utils.quote, но лучше просто передать как есть
        # Сервис emojicdn.elk.sh принимает сам эмодзи в URL
        return f"https://emojicdn.elk.sh/{emoji_char}?style=apple&size={size}"
    except:
        return None

def load_emoji_image(emoji_char, size=160):
    """Загружает PNG эмодзи по URL и возвращает PIL Image (RGBA)."""
    url = f"https://emojicdn.elk.sh/{emoji_char}?style=apple&size={size}"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGBA")
        return img
    except Exception as e:
        st.warning(f"Не удалось загрузить эмодзи: {e}. Использую заглушку.")
        # Заглушка – прозрачное изображение с текстом
        img = Image.new("RGBA", (size, size), (255, 0, 0, 128))
        return img

# -------------------------------------------------------------------
# Загрузка модели YOLO
# -------------------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        model_path = hf_hub_download(
            repo_id="MrBrightSun/face_yolo_26",
            filename="face_best_yolo_26_2ep.pt",
            cache_dir="./model_cache"
        )
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Не удалось загрузить модель: {e}")
        return None

# -------------------------------------------------------------------
# Наложение изображения на лица
# -------------------------------------------------------------------
def overlay_image_on_faces(image_np, boxes, overlay_img):
    """Накладывает overlay_img (RGBA) на каждое лицо в boxes."""
    overlay_np = np.array(overlay_img)
    for (x1, y1, x2, y2) in boxes:
        x1, x2 = sorted([int(x1), int(x2)])
        y1, y2 = sorted([int(y1), int(y2)])
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            continue
        # Масштабируем эмодзи до размера лица
        resized = cv2.resize(overlay_np, (w, h), interpolation=cv2.INTER_AREA)
        alpha = resized[:, :, 3] / 255.0
        for c in range(3):
            image_np[y1:y2, x1:x2, c] = (1 - alpha) * image_np[y1:y2, x1:x2, c] + alpha * resized[:, :, c]
    return image_np

# -------------------------------------------------------------------
# Основная логика
# -------------------------------------------------------------------
def main():
    model = load_model()
    if model is None:
        st.stop()

    # Боковая панель
    st.sidebar.header("Параметры детекции")
    conf_thresh = st.sidebar.slider("Порог уверенности", 0.0, 1.0, 0.5, 0.05)
    iou_thresh = st.sidebar.slider("Порог IoU", 0.0, 1.0, 0.45, 0.05)
    show_boxes = st.sidebar.checkbox("Показывать рамки", value=True)

    st.sidebar.header("Эмодзи-заглушка")
    emoji_text = st.sidebar.text_input("Введите эмодзи", value="🤦‍♂️")
    emoji_size = st.sidebar.slider("Размер эмодзи (исходный, px)", 64, 256, 128, 16)

    # Загружаем картинку эмодзи
    if emoji_text:
        with st.spinner("Загружаем эмодзи..."):
            emoji_img = load_emoji_image(emoji_text, size=emoji_size)
        st.sidebar.image(emoji_img, caption="Выбранный эмодзи", width=100)
    else:
        emoji_img = load_emoji_image("❓", size=emoji_size)
        st.sidebar.warning("Введите эмодзи")

    # Выбор способа загрузки фото
    st.header("Загрузите изображение с лицами")
    input_type = st.radio("Источник:", ["Загрузить файл", "Указать URL"])

    image = None
    if input_type == "Загрузить файл":
        uploaded = st.file_uploader("Выберите файл", type=["jpg", "jpeg", "png"])
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
    else:
        url = st.text_input("Введите URL картинки:")
        if url:
            try:
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                image = Image.open(BytesIO(resp.content)).convert("RGB")
            except Exception as e:
                st.error(f"Не удалось загрузить: {e}")

    if image is not None:
        img_np = np.array(image)
        col1, col2 = st.columns(2)
        col1.subheader("Оригинал")
        col1.image(img_np, use_container_width=True)

        # Детекция
        results = model(img_np, conf=conf_thresh, iou=iou_thresh)
        boxes = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                boxes.append([x1, y1, x2, y2])

        if not boxes:
            st.warning("Лица не обнаружены.")
            result_img = img_np
        else:
            # Накладываем эмодзи
            result_img = overlay_image_on_faces(img_np.copy(), boxes, emoji_img)
            if show_boxes:
                for (x1, y1, x2, y2) in boxes:
                    cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        col2.subheader("Результат")
        col2.image(result_img, use_container_width=True)

        st.subheader("Информация")
        st.write(f"Найдено лиц: {len(boxes)}")
        if boxes:
            for i, box in enumerate(boxes):
                st.write(f"Лицо {i+1}: {box}")

if __name__ == "__main__":
    main()