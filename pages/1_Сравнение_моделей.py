import streamlit as st
from PIL import Image
import os

st.set_page_config(page_title="Обзор моделей", layout="wide")
st.title("📊 Обзор результатов экспериментов")

# Путь к папке с ассетами (предполагается, что она находится в корне проекта)
forest_segmentation_DIR = "forest_segmentation"

# -------------------- Показ предсказаний для 6 моделей на 5 изображениях --------------------
st.header("🔍 Сравнение предсказаний моделей")
st.markdown("Для каждой модели показаны предсказания на одних и тех же 5 изображениях.")

INFERENCE_DIR = os.path.join(forest_segmentation_DIR, "Inference")
if os.path.exists(INFERENCE_DIR):
    # Собираем изображения для каждой модели
    # Предполагается, что файлы названы: 1.png, 2.png, ... для модели 1 и т.д.
    num_models = 6
    num_images = 5
    
    # Создаём заголовки для столбцов: "Оригинал", "Истинная маска", "Модель 1", ..., "Модель 6"
    cols = st.columns(num_models + 2)
    cols[0].markdown("**Оригинал**")
    cols[1].markdown("**Истинная маска**")
    for i in range(num_models):
        cols[i+2].markdown(f"**Модель {i+1}**")
    
    # Для каждого изображения показываем строку
    for img_idx in range(1, num_images + 1):
        # Предполагаем, что оригиналы и истинные маски тоже лежат в Inference?
        # Если нет, пути нужно скорректировать.
        # Здесь для примера предполагается, что оригинал и истинная маска также находятся в Inference.
        orig_path = os.path.join(INFERENCE_DIR, f"orig_{img_idx}.png")
        mask_path = os.path.join(INFERENCE_DIR, f"mask_{img_idx}.png")
        
        if os.path.exists(orig_path) and os.path.exists(mask_path):
            orig_img = Image.open(orig_path)
            mask_img = Image.open(mask_path)
        else:
            # Если файлов нет, создаём заглушку
            orig_img = None
            mask_img = None
        
        cols = st.columns(num_models + 2)
        if orig_img:
            cols[0].image(orig_img, use_container_width=True)
        else:
            cols[0].write("Нет данных")
        if mask_img:
            cols[1].image(mask_img, use_container_width=True)
        else:
            cols[1].write("Нет данных")
        
        for model_idx in range(1, num_models + 1):
            pred_path = os.path.join(INFERENCE_DIR, f"{model_idx}_{img_idx}.png")
            if os.path.exists(pred_path):
                pred_img = Image.open(pred_path)
                cols[model_idx+1].image(pred_img, use_container_width=True)
            else:
                cols[model_idx+1].write("Нет данных")
else:
    st.warning("Папка с предсказаниями не найдена. Поместите изображения в 'forest_segmentation/Inference/'.")

# -------------------- Матрицы ошибок --------------------
st.header("📉 Матрицы ошибок")
MATRIX_DIR = os.path.join(ASSETS_DIR, "Matrix_error")
if os.path.exists(MATRIX_DIR):
    matrix_cols = st.columns(2)
    for model_idx in range(1, 7):
        with matrix_cols[(model_idx-1) % 2]:
            matrix_path = os.path.join(MATRIX_DIR, f"{model_idx}.png")
            if os.path.exists(matrix_path):
                st.image(Image.open(matrix_path), caption=f"Эксперимент {model_idx}", use_container_width=True)
            else:
                st.write(f"Нет матрицы для эксперимента {model_idx}")
else:
    st.warning("Папка с матрицами ошибок не найдена. Поместите изображения в 'forest_segmentation/Matrix_error/'.")

# -------------------- Графики и таблица --------------------
st.header("📈 Графики и таблица метрик")
METRICS_DIR = os.path.join(ASSETS_DIR, "Metrics")
if os.path.exists(METRICS_DIR):
    # Графики для каждой модели
    metrics_cols = st.columns(2)
    for model_idx in range(1, 7):
        with metrics_cols[(model_idx-1) % 2]:
            graph_path = os.path.join(METRICS_DIR, f"{model_idx}.png")
            if os.path.exists(graph_path):
                st.image(Image.open(graph_path), caption=f"График эксперимента {model_idx}", use_container_width=True)
            else:
                st.write(f"Нет графика для эксперимента {model_idx}")
    
    # Сводная таблица
    table_path = os.path.join(METRICS_DIR, "Metrics.png")
    if os.path.exists(table_path):
        st.image(Image.open(table_path), caption="Сводная таблица результатов", use_container_width=True)
    else:
        st.write("Файл сводной таблицы не найден.")
else:
    st.warning("Папка с графиками не найдена. Поместите изображения в 'forest_segmentation/Metrics/'.")