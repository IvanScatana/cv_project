import streamlit as st
from PIL import Image
import os

st.set_page_config(page_title="Обзор моделей", layout="wide")
st.title("📊 Обзор результатов экспериментов")

# -------------------- Информация о датасете --------------------
st.header("📊 Информация о датасете")
st.markdown("""
- **Всего пар изображение-маска:** 5108
- **Тренировочная выборка:** 4086 пар (80%)
- **Валидационная выборка:** 1022 пары (20%)
""")

# Путь к папке с ассетами (корневая папка)
FOREST_SEGMENTATION_DIR = "forest_segmentation"

# -------------------- Показ предсказаний (коллажи для каждой модели) --------------------
st.header("🔍 Сравнение предсказаний моделей")
st.markdown("Для каждой модели показаны предсказания на одних и тех же 5 изображениях.")

INFERENCE_DIR = os.path.join(FOREST_SEGMENTATION_DIR, "Inference")
if os.path.exists(INFERENCE_DIR):
    cols = st.columns(3)
    for model_idx in range(1, 7):
        img_path = os.path.join(INFERENCE_DIR, f"{model_idx}.png")
        if os.path.exists(img_path):
            with cols[(model_idx-1) % 3]:
                st.image(Image.open(img_path), caption=f"Модель {model_idx}", use_container_width=True)
        else:
            st.write(f"Нет данных для модели {model_idx}")
else:
    st.warning("Папка с предсказаниями не найдена. Поместите изображения в 'forest_segmentation/Inference/'.")

# -------------------- Матрицы ошибок --------------------
st.header("📉 Матрицы ошибок")
MATRIX_DIR = os.path.join(FOREST_SEGMENTATION_DIR, "Matrix_error")
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
    st.warning("Папка с матрицами ошибок не найдена.")

# -------------------- Графики и таблица --------------------
st.header("📈 Графики и таблица метрик")
METRICS_DIR = os.path.join(FOREST_SEGMENTATION_DIR, "Metrics")
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
    st.warning("Папка с графиками не найдена.")