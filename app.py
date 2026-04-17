import streamlit as st

st.set_page_config(
    page_title="CV Project",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("Навигация")
st.sidebar.markdown("Выберите страницу в боковом меню.")

st.title("🖥️ Компьютерное зрение")
st.markdown(
    """
    Добро пожаловать!  
    Здесь представлены результаты экспериментов по семантической сегментации лесных массивов.
    """
)