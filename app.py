import streamlit as st
import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import tempfile
import os
from ultralytics import YOLO
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ==================== НАСТРОЙКА СТРАНИЦЫ ====================
st.set_page_config(
    page_title="Детекция лиц YOLOv8",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== ЗАГОЛОВОК ====================
st.title("👤 Детекция лиц с помощью YOLOv8")
st.markdown("---")

# ==================== САЙДБАР НАСТРОЕК ====================
with st.sidebar:
    st.header("⚙️ Настройки")
    
    # Загрузка модели
    st.subheader("Модель")
    model_option = st.radio(
        "Выберите модель:",
        ["Стандартная YOLOv8", "Своя модель"],
        help="Стандартная модель скачается автоматически"
    )
    
    if model_option == "Стандартная YOLOv8":
        model_type = st.selectbox(
            "Тип модели:",
            ["yolov8n.pt (нано, быстрая)", "yolov8s.pt (малая)", "yolov8m.pt (средняя)"],
            index=0
        )
        model_path = model_type.split(" ")[0]  # Извлекаем 'yolov8n.pt'
    else:
        model_path = st.text_input(
            "Путь к вашей модели (weights):",
            value="best.pt",
            help="Укажите путь к файлу .pt с весами модели"
        )
    
    # Параметры детекции
    st.subheader("Параметры детекции")
    confidence_threshold = st.slider(
        "Порог уверенности:",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Минимальная уверенность для детекции лица"
    )
    
    iou_threshold = st.slider(
        "IOU порог (для NMS):",
        min_value=0.0,
        max_value=1.0,
        value=0.45,
        step=0.05,
        help="Порог для подавления немаксимумов"
    )
    
    # Настройки отображения
    st.subheader("Настройки отображения")
    bbox_color = st.color_picker(
        "Цвет bounding box:",
        "#FF0000"
    )
    
    line_thickness = st.slider(
        "Толщина линии:",
        min_value=1,
        max_value=10,
        value=3
    )
    
    show_labels = st.checkbox("Показывать метки", value=True)
    show_conf = st.checkbox("Показывать уверенность", value=True)
    
    if show_labels:
        font_size = st.slider(
            "Размер шрифта:",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.1
        )
    
    # Информация
    st.markdown("---")
    st.info("""
    ### ℹ️ Инструкция:
    1. Загрузите изображение через вкладку **"📤 Загрузка"**
    2. Или вставьте URL через вкладку **"🔗 URL"**
    3. Нажмите **"🚀 Запустить детекцию"**
    4. Просмотрите результаты во вкладке **"📊 Результаты"**
    """)
    
    st.markdown("---")
    st.caption(f"Время: {datetime.now().strftime('%H:%M:%S')}")

# ==================== ИНИЦИАЛИЗАЦИЯ СЕССИИ ====================
if 'model' not in st.session_state:
    st.session_state.model = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None

# ==================== ФУНКЦИИ ====================
@st.cache_resource
def load_model(model_path):
    """Загрузка модели YOLOv8"""
    try:
        with st.spinner(f"Загрузка модели {model_path}..."):
            model = YOLO(model_path)
            st.success(f"✅ Модель {model_path} загружена")
            return model
    except Exception as e:
        st.error(f"❌ Ошибка загрузки модели: {e}")
        return None

def hex_to_bgr(hex_color):
    """Конвертация HEX цвета в BGR для OpenCV"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))

def draw_detections(image_np, detections, bbox_color, line_thickness, show_labels, show_conf, font_size):
    """Отрисовка bounding boxes на изображении"""
    img_copy = image_np.copy()
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        conf = det['confidence']
        
        # Рисование bounding box
        color = hex_to_bgr(bbox_color)
        cv2.rectangle(
            img_copy,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            color,
            line_thickness
        )
        
        # Добавление текста
        if show_labels:
            label = f"Face: {conf:.2f}" if show_conf else "Face"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = font_size
            thickness = max(1, line_thickness // 2)
            
            # Размер текста для фона
            (text_width, text_height), _ = cv2.getTextSize(
                label, font, font_scale, thickness
            )
            
            # Фон для текста
            cv2.rectangle(
                img_copy,
                (int(x1), int(y1) - text_height - 10),
                (int(x1) + text_width, int(y1)),
                color,
                -1
            )
            
            # Текст
            cv2.putText(
                img_copy,
                label,
                (int(x1), int(y1) - 5),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA
            )
    
    return img_copy

def process_image(model, image, conf_threshold, iou_threshold):
    """Обработка изображения и детекция лиц"""
    try:
        # Конвертация в numpy array
        if isinstance(image, Image.Image):
            image_np = np.array(image)
            # Конвертация RGB в BGR для OpenCV
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            image_np = image.copy()
        
        original_height, original_width = image_np.shape[:2]
        
        # Детекция
        results = model(
            image_np, 
            conf=conf_threshold, 
            iou=iou_threshold,
            verbose=False,
            classes=[0]  # 0 - класс 'person' в COCO, для лиц используйте свою модель
        )
        
        # Извлечение детекций
        detections = []
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(conf),
                        'class': cls,
                        'class_name': model.names[cls] if hasattr(model, 'names') else 'face',
                        'area': (x2 - x1) * (y2 - y1),
                        'width': x2 - x1,
                        'height': y2 - y1
                    })
        
        return detections, image_np, original_width, original_height
        
    except Exception as e:
        st.error(f"❌ Ошибка обработки изображения: {e}")
        return [], None, 0, 0

def calculate_metrics(detections):
    """Расчет метрик детекции"""
    if not detections:
        return None
    
    confidences = [d['confidence'] for d in detections]
    areas = [d['area'] for d in detections]
    widths = [d['width'] for d in detections]
    heights = [d['height'] for d in detections]
    
    metrics = {
        'total_faces': len(detections),
        'avg_confidence': np.mean(confidences),
        'max_confidence': np.max(confidences),
        'min_confidence': np.min(confidences),
        'confidence_std': np.std(confidences) if len(confidences) > 1 else 0,
        'avg_area': np.mean(areas),
        'avg_width': np.mean(widths),
        'avg_height': np.mean(heights),
        'total_area': np.sum(areas),
        'detection_time': datetime.now().strftime("%H:%M:%S")
    }
    
    # Распределение по уверенности
    bins = [0.0, 0.3, 0.5, 0.7, 0.9, 1.01]
    bin_labels = ['0-0.3', '0.3-0.5', '0.5-0.7', '0.7-0.9', '0.9-1.0']
    bin_counts = []
    
    for i in range(len(bins)-1):
        count = len([c for c in confidences if bins[i] <= c < bins[i+1]])
        bin_counts.append(count)
    
    metrics['confidence_distribution'] = {
        'labels': bin_labels,
        'counts': bin_counts
    }
    
    return metrics

# ==================== ОСНОВНОЕ ПРИЛОЖЕНИЕ ====================
tab1, tab2, tab3 = st.tabs(["📤 Загрузка изображения", "🔗 URL изображения", "📊 Результаты"])

# Вкладка 1: Загрузка изображения
with tab1:
    st.header("📤 Загрузка изображения с устройства")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Выберите изображение:",
            type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
            help="Поддерживаемые форматы: JPG, PNG, BMP, WebP",
            key="uploader_1"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.session_state.original_image = image
            
            st.image(
                image, 
                caption=f"📏 Размер: {image.size[0]}x{image.size[1]} пикселей",
                use_column_width=True
            )
            
            if st.button("🚀 Запустить детекцию", type="primary", use_container_width=True):
                # Загрузка модели
                if st.session_state.model is None or True:  # Всегда перезагружаем для новой модели
                    st.session_state.model = load_model(model_path)
                
                if st.session_state.model:
                    with st.spinner("🔍 Выполняется детекция лиц..."):
                        # Обработка изображения
                        detections, image_np, width, height = process_image(
                            st.session_state.model, 
                            image, 
                            confidence_threshold,
                            iou_threshold
                        )
                        
                        if image_np is not None:
                            # Отрисовка детекций
                            processed_img = draw_detections(
                                image_np, 
                                detections, 
                                bbox_color, 
                                line_thickness, 
                                show_labels, 
                                show_conf,
                                font_size if show_labels else 1.0
                            )
                            
                            # Конвертация BGR в RGB для отображения
                            processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                            st.session_state.processed_image = Image.fromarray(processed_img_rgb)
                            st.session_state.results = {
                                'detections': detections,
                                'original_size': (width, height),
                                'metrics': calculate_metrics(detections)
                            }
                            
                            st.success(f"✅ Обнаружено лиц: {len(detections)}")
    
    with col2:
        if st.session_state.processed_image is not None and st.session_state.results:
            st.header("📊 Результаты детекции")
            
            # Отображение обработанного изображения
            st.image(
                st.session_state.processed_image,
                caption=f"👥 Обнаружено лиц: {len(st.session_state.results['detections'])}",
                use_column_width=True
            )
            
            # Быстрая статистика
            if st.session_state.results['detections']:
                detections = st.session_state.results['detections']
                with st.expander("📈 Быстрая статистика", expanded=True):
                    cols = st.columns(4)
                    cols[0].metric("Лица", len(detections))
                    cols[1].metric("Ср. уверенность", f"{np.mean([d['confidence'] for d in detections]):.1%}")
                    cols[2].metric("Мин. уверенность", f"{np.min([d['confidence'] for d in detections]):.1%}")
                    cols[3].metric("Макс. уверенность", f"{np.max([d['confidence'] for d in detections]):.1%}")

# Вкладка 2: URL изображения
with tab2:
    st.header("🔗 Загрузка изображения по URL")
    
    url = st.text_input(
        "Введите URL изображения:",
        placeholder="https://example.com/image.jpg",
        help="Введите полный URL изображения (поддерживаются JPG, PNG)"
    )
    
    if url:
        try:
            response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                st.session_state.original_image = image
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.image(
                        image,
                        caption=f"📏 Размер: {image.size[0]}x{image.size[1]} пикселей",
                        use_column_width=True
                    )
                    
                    if st.button("🚀 Запустить детекцию из URL", type="primary", use_container_width=True):
                        # Загрузка модели
                        if st.session_state.model is None or True:
                            st.session_state.model = load_model(model_path)
                        
                        if st.session_state.model:
                            with st.spinner("🔍 Выполняется детекция лиц..."):
                                # Обработка изображения
                                detections, image_np, width, height = process_image(
                                    st.session_state.model, 
                                    image, 
                                    confidence_threshold,
                                    iou_threshold
                                )
                                
                                if image_np is not None:
                                    # Отрисовка детекций
                                    processed_img = draw_detections(
                                        image_np, 
                                        detections, 
                                        bbox_color, 
                                        line_thickness, 
                                        show_labels, 
                                        show_conf,
                                        font_size if show_labels else 1.0
                                    )
                                    
                                    # Конвертация BGR в RGB для отображения
                                    processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                                    st.session_state.processed_image = Image.fromarray(processed_img_rgb)
                                    st.session_state.results = {
                                        'detections': detections,
                                        'original_size': (width, height),
                                        'metrics': calculate_metrics(detections)
                                    }
                                    
                                    st.success(f"✅ Обнаружено лиц: {len(detections)}")
                
                with col2:
                    if st.session_state.processed_image is not None and st.session_state.results:
                        st.header("📊 Результаты детекции")
                        
                        # Отображение обработанного изображения
                        st.image(
                            st.session_state.processed_image,
                            caption=f"👥 Обнаружено лиц: {len(st.session_state.results['detections'])}",
                            use_column_width=True
                        )
            else:
                st.error(f"❌ Ошибка загрузки изображения. Код: {response.status_code}")
        except Exception as e:
            st.error(f"❌ Ошибка: {str(e)}")

# Вкладка 3: Результаты и метрики
with tab3:
    st.header("📊 Детальный анализ результатов")
    
    if st.session_state.results is not None and st.session_state.results['detections']:
        detections = st.session_state.results['detections']
        metrics = st.session_state.results['metrics']
        
        # Основные метрики
        st.subheader("📈 Основные метрики")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Общее количество лиц", metrics['total_faces'])
        
        with col2:
            st.metric("Средняя уверенность", f"{metrics['avg_confidence']:.1%}")
        
        with col3:
            st.metric("Максимальная уверенность", f"{metrics['max_confidence']:.1%}")
        
        with col4:
            st.metric("Минимальная уверенность", f"{metrics['min_confidence']:.1%}")
        
        st.markdown("---")
        
        # Детальная таблица
        st.subheader("📋 Детекции")
        df_detections = pd.DataFrame(detections)
        
        # Форматирование данных
        df_display = df_detections.copy()
        df_display['confidence'] = df_display['confidence'].apply(lambda x: f"{x:.1%}")
        df_display['area'] = df_display['area'].apply(lambda x: f"{int(x)} px²")
        df_display['width'] = df_display['width'].apply(lambda x: f"{int(x)} px")
        df_display['height'] = df_display['height'].apply(lambda x: f"{int(x)} px")
        
        # Переименование колонок
        df_display = df_display.rename(columns={
            'class_name': 'Класс',
            'confidence': 'Уверенность',
            'area': 'Площадь',
            'width': 'Ширина',
            'height': 'Высота'
        })
        
        st.dataframe(
            df_display[['Класс', 'Уверенность', 'Площадь', 'Ширина', 'Высота']],
            use_container_width=True,
            hide_index=True
        )
        
        st.markdown("---")
        
        # Графики
        st.subheader("📊 Визуализация данных")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Гистограмма уверенности
            fig_conf = px.histogram(
                df_detections,
                x='confidence',
                nbins=20,
                title="Распределение уверенности",
                labels={'confidence': 'Уверенность', 'count': 'Количество'},
                color_discrete_sequence=['#FF4B4B']
            )
            fig_conf.update_layout(xaxis_range=[0, 1], bargap=0.1)
            st.plotly_chart(fig_conf, use_container_width=True)
        
        with col2:
            # Круговая диаграмма распределения
            if metrics['confidence_distribution']['counts']:
                fig_pie = px.pie(
                    values=metrics['confidence_distribution']['counts'],
                    names=metrics['confidence_distribution']['labels'],
                    title="Распределение по диапазонам уверенности",
                    color_discrete_sequence=px.colors.sequential.RdBu
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        
        # График соотношения сторон
        st.subheader("📐 Анализ bounding boxes")
        
        if len(detections) > 1:
            df_detections['aspect_ratio'] = df_detections['width'] / df_detections['height']
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_box = px.box(
                    df_detections,
                    y='aspect_ratio',
                    title="Распределение соотношений сторон (ширина/высота)",
                    labels={'aspect_ratio': 'Соотношение сторон'},
                    color_discrete_sequence=['#00CC96']
                )
                st.plotly_chart(fig_box, use_container_width=True)
            
            with col2:
                fig_scatter = px.scatter(
                    df_detections,
                    x='width',
                    y='height',
                    size='area',
                    color='confidence',
                    title="Размеры bounding boxes",
                    labels={'width': 'Ширина (px)', 'height': 'Высота (px)', 'confidence': 'Уверенность'},
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
        
        st.markdown("---")
        
        # Экспорт результатов
        st.subheader("💾 Экспорт результатов")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Скачать изображение", use_container_width=True):
                if st.session_state.processed_image:
                    # Создание временного файла
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                        st.session_state.processed_image.save(tmp_file.name, 'JPEG', quality=95)
                        
                        with open(tmp_file.name, 'rb') as f:
                            st.download_button(
                                label="Нажмите для скачивания",
                                data=f,
                                file_name="detected_faces.jpg",
                                mime="image/jpeg",
                                key="download_img"
                            )
        
        with col2:
            if st.button("📊 Скачать данные (CSV)", use_container_width=True):
                csv = df_detections.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Нажмите для скачивания",
                    data=csv,
                    file_name="face_detection_data.csv",
                    mime="text/csv",
                    key="download_csv"
                )
        
        with col3:
            if st.button("🔄 Сбросить результаты", use_container_width=True):
                st.session_state.results = None
                st.session_state.processed_image = None
                st.rerun()
    
    elif st.session_state.results is not None and len(st.session_state.results['detections']) == 0:
        st.warning("⚠️ Лица не обнаружены. Попробуйте:")
        st.markdown("""
        1. Уменьшить **порог уверенности** в настройках
        2. Загрузить другое изображение
        3. Проверить, что на изображении есть лица
        """)
    else:
        st.info("👈 Загрузите изображение и выполните детекцию, чтобы увидеть результаты")

# ==================== ФУТЕР ====================
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)
with footer_col1:
    st.caption("🛠️ Детекция лиц с YOLOv8")
with footer_col2:
    st.caption(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
with footer_col3:
    st.caption("📊 Streamlit + Ultralytics")

# ==================== СТИЛИ CSS ====================
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    .css-1d391kg {
        padding: 1rem;
        border-radius: 10px;
        background-color: #f8f9fa;
    }
    h1, h2, h3 {
        color: #1E3A8A;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)