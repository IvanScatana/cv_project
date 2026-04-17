import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import requests
from io import BytesIO

st.set_page_config(page_title="Face Blurring App", layout="wide")
st.title("Размытие лиц с помощью YOLO")

@st.cache_resource
def load_model():
    """Загрузка модели YOLO для детекции лиц с Hugging Face Hub."""
    try:
        model_path = hf_hub_download(
            repo_id="MrBrightSun/face_yolo_26",
            filename="face_best_yolo_26_2ep.pt",
            cache_dir="./model_cache"
        )
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Не удалось загрузить модель с Hugging Face Hub: {e}")
        st.info("Проверьте репозиторий и имя файла.")
        return None

def blur_faces(image, boxes, blur_strength=51):
    """
    Применяет размытие по Гауссу к областям, заданным ограничивающими рамками.

    Args:
        image (np.ndarray): Входное изображение в формате BGR.
        boxes (list): Список рамок [x1, y1, x2, y2].
        blur_strength (int): Размер ядра размытия (нечётное число).

    Returns:
        np.ndarray: Изображение с размытыми лицами.
    """
    if blur_strength % 2 == 0:
        blur_strength += 1
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        face_region = image[y1:y2, x1:x2]
        if face_region.size == 0:
            continue
        blurred_face = cv2.GaussianBlur(face_region, (blur_strength, blur_strength), 0)
        image[y1:y2, x1:x2] = blurred_face
    return image

def load_image_from_url(url):
    """Загружает изображение по URL и возвращает PIL Image."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return img
    except Exception as e:
        st.error(f"Ошибка загрузки изображения по ссылке: {e}")
        return None

def process_image(image_np, model, confidence_threshold, iou_threshold, blur_strength, show_boxes):
    """Выполняет детекцию, размытие и возвращает результат и информацию."""
    results = model(image_np, conf=confidence_threshold, iou=iou_threshold)
    boxes = []
    if results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            boxes.append([x1, y1, x2, y2])

    if boxes:
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        image_blurred_bgr = blur_faces(image_bgr, boxes, blur_strength)
        image_blurred = cv2.cvtColor(image_blurred_bgr, cv2.COLOR_BGR2RGB)
    else:
        st.warning("Лица не обнаружены.")
        image_blurred = image_np

    if show_boxes and boxes:
        image_with_boxes = image_blurred.copy()
        for box in boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
        result_img = image_with_boxes
        caption = "Размытие с рамками"
    else:
        result_img = image_blurred
        caption = "Размытое изображение"

    return result_img, caption, boxes

def main():
    model = load_model()
    if model is None:
        st.stop()

    st.sidebar.header("Параметры детекции")
    confidence_threshold = st.sidebar.slider(
        "Порог уверенности", min_value=0.0, max_value=1.0, value=0.5, step=0.05
    )
    iou_threshold = st.sidebar.slider(
        "Порог IoU", min_value=0.0, max_value=1.0, value=0.45, step=0.05
    )
    blur_strength = st.sidebar.slider(
        "Сила размытия", min_value=1, max_value=99, value=51, step=2
    )
    show_boxes = st.sidebar.checkbox("Показывать рамки детекции", value=True)

    st.header("Загрузите изображение")
    input_type = st.radio("Выберите способ загрузки:", ["Загрузить файл", "Указать URL"])

    image = None
    if input_type == "Загрузить файл":
        uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
    else:
        url = st.text_input("Введите URL изображения:")
        if url:
            with st.spinner("Загрузка изображения..."):
                image = load_image_from_url(url)

    if image is not None:
        image_np = np.array(image)
        st.subheader("Оригинальное изображение")
        st.image(image_np, caption="Оригинал", use_container_width=True)

        with st.spinner("Обработка..."):
            result_img, caption, boxes = process_image(
                image_np, model, confidence_threshold, iou_threshold,
                blur_strength, show_boxes
            )

        st.subheader("Результат")
        st.image(result_img, caption=caption, use_container_width=True)

        st.subheader("Информация о детекции")
        st.write(f"Обнаружено лиц: {len(boxes)}")
        if boxes:
            st.write("Координаты рамок (x1, y1, x2, y2):")
            for i, box in enumerate(boxes):
                st.write(f"Лицо {i+1}: {box}")

if __name__ == "__main__":
    main()