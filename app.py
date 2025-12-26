import streamlit as st
import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import tempfile
import os
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Настройка страницы
st.set_page_config(
    page_title="Детекция лиц YOLOv8",
    page_icon="👤",
    layout="wide"
)

# Заголовок приложения
st.title("👤 Детекция лиц с помощью YOLOv8")
st.markdown("---")

# Сайдбар для настроек
with st.sidebar:
    st.header("⚙️ Настройки")
    
    # Загрузка модели
    model_path = st.text_input(
        "Путь к модели YOLOv8",
        value="/content/yolov8n.pt",
        help="Укажите путь к файлу модели .pt"
    )
    
    # Порог уверенности
    confidence_threshold = st.slider(
        "Порог уверенности",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Минимальная уверенность для детекции"
    )
    
    # Цвет bounding box
    bbox_color = st.color_picker(
        "Цвет bounding box",
        "#FF0000"
    )
    
    # Толщина линии
    line_thickness = st.slider(
        "Толщина линии",
        min_value=1,
        max_value=10,
        value=3
    )
    
    # Размер шрифта
    font_size = st.slider(
        "Размер шрифта",
        min_value=0.5,
        max_value=3.0,
        value=1.0,
        step=0.1
    )
    
    st.markdown("---")
    st.info("""
    ### Инструкция:
    1. Загрузите изображение через вкладку "Загрузка"
    2. Или вставьте URL изображения через вкладку "URL"
    3. Нажмите "Запустить детекцию"
    4. Просмотрите результаты и метрики
    """)

# Основное содержимое
tab1, tab2, tab3 = st.tabs(["📤 Загрузка изображения", "🔗 URL изображения", "📊 Метрики и анализ"])

# Инициализация состояния сессии
if 'results' not in st.session_state:
    st.session_state.results = None
if 'image' not in st.session_state:
    st.session_state.image = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None

def load_model(model_path):
    """Загрузка модели YOLOv8"""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        return None

def process_image(model, image, conf_threshold):
    """Обработка изображения и детекция лиц"""
    try:
        # Конвертация цвета для OpenCV
        if isinstance(image, Image.Image):
            image_np = np.array(image)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            image_np = image.copy()
        
        # Выполнение детекции
        results = model(image_np, conf=conf_threshold, verbose=False)
        
        # Извлечение предсказаний
        predictions = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    predictions.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(conf),
                        'class': cls,
                        'class_name': model.names[cls]
                    })
                    
                    # Рисование bounding box
                    cv2.rectangle(
                        image_np,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        tuple(int(bbox_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)),
                        line_thickness
                    )
                    
                    # Добавление текста с уверенностью
                    label = f"Face: {conf:.2f}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = font_size
                    thickness = max(1, line_thickness // 2)
                    
                    # Получение размера текста для фона
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, font, font_scale, thickness
                    )
                    
                    # Рисование фона для текста
                    cv2.rectangle(
                        image_np,
                        (int(x1), int(y1) - text_height - 10),
                        (int(x1) + text_width, int(y1)),
                        tuple(int(bbox_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)),
                        -1
                    )
                    
                    # Добавление текста
                    cv2.putText(
                        image_np,
                        label,
                        (int(x1), int(y1) - 5),
                        font,
                        font_scale,
                        (255, 255, 255),
                        thickness,
                        cv2.LINE_AA
                    )
        
        # Конвертация обратно в RGB для отображения
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        
        return predictions, image_rgb, results[0]
        
    except Exception as e:
        st.error(f"Ошибка обработки изображения: {e}")
        return [], None, None

def calculate_metrics(predictions):
    """Расчет метрик"""
    if not predictions:
        return None
    
    metrics = {
        'total_faces': len(predictions),
        'avg_confidence': np.mean([p['confidence'] for p in predictions]),
        'max_confidence': np.max([p['confidence'] for p in predictions]) if predictions else 0,
        'min_confidence': np.min([p['confidence'] for p in predictions]) if predictions else 0,
        'confidence_std': np.std([p['confidence'] for p in predictions]) if len(predictions) > 1 else 0
    }
    
    # Распределение по уверенности
    confidence_bins = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
    conf_counts = []
    conf_labels = []
    
    for i in range(len(confidence_bins)-1):
        count = len([p for p in predictions if confidence_bins[i] <= p['confidence'] < confidence_bins[i+1]])
        if count > 0:
            conf_counts.append(count)
            conf_labels.append(f"{confidence_bins[i]:.1f}-{confidence_bins[i+1]:.1f}")
    
    metrics['confidence_distribution'] = {
        'bins': conf_labels,
        'counts': conf_counts
    }
    
    # Размеры bounding boxes
    if predictions:
        bbox_areas = []
        bbox_widths = []
        bbox_heights = []
        
        for p in predictions:
            x1, y1, x2, y2 = p['bbox']
            width = x2 - x1
            height = y2 - y1
            area = width * height
            
            bbox_widths.append(width)
            bbox_heights.append(height)
            bbox_areas.append(area)
        
        metrics['bbox_stats'] = {
            'avg_area': np.mean(bbox_areas),
            'avg_width': np.mean(bbox_widths),
            'avg_height': np.mean(bbox_heights),
            'min_area': np.min(bbox_areas),
            'max_area': np.max(bbox_areas)
        }
    
    return metrics

# Вкладка загрузки изображения
with tab1:
    st.header("Загрузите изображение для детекции лиц")
    
    uploaded_file = st.file_uploader(
        "Выберите изображение",
        type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
        help="Поддерживаемые форматы: JPG, PNG, BMP, WebP"
    )
    
    if uploaded_file is not None:
        # Загрузка изображения
        image = Image.open(uploaded_file)
        st.session_state.image = image
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Исходное изображение")
            st.image(image, caption="Загруженное изображение", use_column_width=True)
            
            st.info(f"Размер изображения: {image.size[0]}x{image.size[1]} пикселей")
        
        with col2:
            st.subheader("Обработанное изображение")
            
            if st.button("🚀 Запустить детекцию", type="primary", use_container_width=True):
                with st.spinner("Выполняется детекция лиц..."):
                    # Загрузка модели
                    model = load_model(model_path)
                    
                    if model:
                        # Обработка изображения
                        predictions, processed_image, result_obj = process_image(
                            model, image, confidence_threshold
                        )
                        
                        if processed_image is not None:
                            # Сохранение результатов
                            st.session_state.results = {
                                'predictions': predictions,
                                'processed_image': processed_image,
                                'result_obj': result_obj
                            }
                            
                            # Расчет метрик
                            st.session_state.metrics = calculate_metrics(predictions)
                            
                            # Отображение результата
                            st.image(
                                processed_image,
                                caption=f"Обнаружено лиц: {len(predictions)}",
                                use_column_width=True
                            )
                            
                            # Отображение информации о детекциях
                            if predictions:
                                st.success(f"✅ Обнаружено {len(predictions)} лиц")
                                
                                # Таблица с результатами
                                df_predictions = pd.DataFrame(predictions)
                                df_predictions['confidence_percent'] = df_predictions['confidence'] * 100
                                df_predictions = df_predictions[['class_name', 'confidence_percent']]
                                df_predictions.columns = ['Класс', 'Уверенность (%)']
                                
                                st.dataframe(
                                    df_predictions.style.format({'Уверенность (%)': '{:.2f}%'}),
                                    use_container_width=True
                                )
                            else:
                                st.warning("⚠️ Лица не обнаружены")

# Вкладка URL изображения
with tab2:
    st.header("Вставьте URL изображения")
    
    url = st.text_input(
        "URL изображения",
        placeholder="https://example.com/image.jpg",
        help="Введите полный URL изображения"
    )
    
    if url:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                st.session_state.image = image
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Исходное изображение")
                    st.image(image, caption="Изображение по URL", use_column_width=True)
                    st.info(f"Размер изображения: {image.size[0]}x{image.size[1]} пикселей")
                
                with col2:
                    st.subheader("Обработанное изображение")
                    
                    if st.button("🚀 Запустить детекцию из URL", type="primary", use_container_width=True):
                        with st.spinner("Выполняется детекция лиц..."):
                            # Загрузка модели
                            model = load_model(model_path)
                            
                            if model:
                                # Обработка изображения
                                predictions, processed_image, result_obj = process_image(
                                    model, image, confidence_threshold
                                )
                                
                                if processed_image is not None:
                                    # Сохранение результатов
                                    st.session_state.results = {
                                        'predictions': predictions,
                                        'processed_image': processed_image,
                                        'result_obj': result_obj
                                    }
                                    
                                    # Расчет метрик
                                    st.session_state.metrics = calculate_metrics(predictions)
                                    
                                    # Отображение результата
                                    st.image(
                                        processed_image,
                                        caption=f"Обнаружено лиц: {len(predictions)}",
                                        use_column_width=True
                                    )
                                    
                                    # Отображение информации о детекциях
                                    if predictions:
                                        st.success(f"✅ Обнаружено {len(predictions)} лиц")
                                        
                                        # Таблица с результатами
                                        df_predictions = pd.DataFrame(predictions)
                                        df_predictions['confidence_percent'] = df_predictions['confidence'] * 100
                                        df_predictions = df_predictions[['class_name', 'confidence_percent']]
                                        df_predictions.columns = ['Класс', 'Уверенность (%)']
                                        
                                        st.dataframe(
                                            df_predictions.style.format({'Уверенность (%)': '{:.2f}%'}),
                                            use_container_width=True
                                        )
                                    else:
                                        st.warning("⚠️ Лица не обнаружены")
            else:
                st.error(f"Ошибка загрузки изображения. Код: {response.status_code}")
        except Exception as e:
            st.error(f"Ошибка: {e}")

# Вкладка метрик и анализа
with tab3:
    st.header("📊 Метрики и анализ результатов")
    
    if st.session_state.metrics is not None and st.session_state.results is not None:
        metrics = st.session_state.metrics
        predictions = st.session_state.results['predictions']
        
        if predictions:
            # Основные метрики
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="Общее количество лиц",
                    value=metrics['total_faces']
                )
            
            with col2:
                st.metric(
                    label="Средняя уверенность",
                    value=f"{metrics['avg_confidence']:.2%}"
                )
            
            with col3:
                st.metric(
                    label="Максимальная уверенность",
                    value=f"{metrics['max_confidence']:.2%}"
                )
            
            with col4:
                st.metric(
                    label="Минимальная уверенность",
                    value=f"{metrics['min_confidence']:.2%}"
                )
            
            st.markdown("---")
            
            # Графики
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Распределение уверенности")
                
                # Подготовка данных для графика
                conf_values = [p['confidence'] for p in predictions]
                
                fig1 = px.histogram(
                    x=conf_values,
                    nbins=20,
                    title="Гистограмма уверенности",
                    labels={'x': 'Уверенность', 'y': 'Количество'},
                    color_discrete_sequence=['#FF4B4B']
                )
                
                fig1.update_layout(
                    xaxis_range=[0, 1],
                    bargap=0.1
                )
                
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                st.subheader("Диаграмма распределения по диапазонам уверенности")
                
                if 'confidence_distribution' in metrics:
                    fig2 = px.pie(
                        values=metrics['confidence_distribution']['counts'],
                        names=metrics['confidence_distribution']['bins'],
                        title="Распределение по диапазонам уверенности",
                        color_discrete_sequence=px.colors.sequential.RdBu
                    )
                    
                    st.plotly_chart(fig2, use_container_width=True)
            
            st.markdown("---")
            
            # Дополнительные метрики
            st.subheader("Статистика bounding boxes")
            
            if 'bbox_stats' in metrics:
                bbox_stats = metrics['bbox_stats']
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="Средняя площадь",
                        value=f"{bbox_stats['avg_area']:.0f} px²"
                    )
                
                with col2:
                    st.metric(
                        label="Средняя ширина",
                        value=f"{bbox_stats['avg_width']:.0f} px"
                    )
                
                with col3:
                    st.metric(
                        label="Средняя высота",
                        value=f"{bbox_stats['avg_height']:.0f} px"
                    )
                
                # График соотношения сторон
                st.subheader("Соотношение сторон bounding boxes")
                
                aspect_ratios = []
                for p in predictions:
                    x1, y1, x2, y2 = p['bbox']
                    width = x2 - x1
                    height = y2 - y1
                    if height > 0:
                        aspect_ratios.append(width / height)
                
                if aspect_ratios:
                    fig3 = px.box(
                        y=aspect_ratios,
                        title="Распределение соотношений сторон (ширина/высота)",
                        labels={'y': 'Соотношение сторон'},
                        color_discrete_sequence=['#00CC96']
                    )
                    
                    st.plotly_chart(fig3, use_container_width=True)
            
            # Кнопка для скачивания результатов
            st.markdown("---")
            st.subheader("Экспорт результатов")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📥 Скачать обработанное изображение"):
                    # Конвертация в формат для скачивания
                    processed_img_pil = Image.fromarray(st.session_state.results['processed_image'])
                    
                    # Сохранение во временный файл
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                        processed_img_pil.save(tmp_file.name, format='JPEG', quality=95)
                        
                        with open(tmp_file.name, 'rb') as file:
                            btn = st.download_button(
                                label="Нажмите для скачивания",
                                data=file,
                                file_name="detected_faces.jpg",
                                mime="image/jpeg"
                            )
            
            with col2:
                if st.button("📊 Скачать метрики в CSV"):
                    # Подготовка данных для CSV
                    csv_data = []
                    for i, pred in enumerate(predictions):
                        csv_data.append({
                            'ID': i+1,
                            'Class': pred['class_name'],
                            'Confidence': pred['confidence'],
                            'Confidence_%': pred['confidence'] * 100,
                            'X1': pred['bbox'][0],
                            'Y1': pred['bbox'][1],
                            'X2': pred['bbox'][2],
                            'Y2': pred['bbox'][3],
                            'Width': pred['bbox'][2] - pred['bbox'][0],
                            'Height': pred['bbox'][3] - pred['bbox'][1],
                            'Area': (pred['bbox'][2] - pred['bbox'][0]) * (pred['bbox'][3] - pred['bbox'][1])
                        })
                    
                    df_csv = pd.DataFrame(csv_data)
                    
                    # Создание CSV
                    csv_string = df_csv.to_csv(index=False).encode('utf-8')
                    
                    st.download_button(
                        label="Нажмите для скачивания CSV",
                        data=csv_string,
                        file_name="face_detection_metrics.csv",
                        mime="text/csv"
                    )
        
        else:
            st.warning("Лица не обнаружены. Нет данных для анализа.")
    else:
        st.info("Загрузите изображение и выполните детекцию, чтобы увидеть метрики.")

# Футер
st.markdown("---")
st.caption("Детекция лиц с использованием YOLOv8 | Streamlit приложение")