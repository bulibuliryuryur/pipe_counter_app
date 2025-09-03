import streamlit as st
from PIL import Image
import numpy as np
import torch
from streamlit_cropper import st_cropper

st.set_page_config(page_title="パイプカウントアプリ(YOLOv5)", layout="wide")
st.title("パイプの本数を数えるWebアプリ(YOLOv5版)")

# --- YOLOv5モデルのロード ---
@st.cache_resource
def load_yolo_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
    return model

model = load_yolo_model()

# --- 画像アップロード ---
uploaded_file = st.file_uploader("パイプの画像を選択してください", type=["jpg", "jpeg", "png"])

if uploaded_file:
    original_image = Image.open(uploaded_file)
    st.image(original_image, caption="元の画像", use_column_width=True)

    st.write("---")
    st.subheader("画像上でトリミングしてください")
    st.info("赤い枠をドラッグして範囲を指定すると自動的に解析されます。")

    # --- streamlit-cropper 設定 ---
    cropped_image = st_cropper(
        original_image,
        realtime_update=True,
        box_color="#FF0000",
        box_thickness=3,
        aspect_ratio=None,   # 自由な比率
        return_type="array"  # NumPy配列で取得
    )

    # --- トリミング後に自動解析 ---
    if cropped_image is not None:
        st.write("---")
        st.subheader("トリミング後の画像")
        st.image(cropped_image, caption="トリミングされた画像", use_column_width=True)

        img_rgb = cropped_image.copy()  # RGB形式
        with st.spinner("解析中..."):
            results = model(img_rgb)

        # --- YOLO 検出結果表示 ---
        annotated_frame = results.render()[0]
        st.image(annotated_frame, caption="検出結果", use_column_width=True)

        detections = results.pred[0]
        num_pipes = len(detections)
        st.success(f"検出されたパイプの数: **{num_pipes}本**")

        st.subheader("個別パイプ画像")
        if num_pipes > 0:
            cols = st.columns(min(num_pipes, 5))
            for i, box in enumerate(detections):
                x1, y1, x2, y2 = map(int, box[:4])
                final_cropped_pipe = img_rgb[y1:y2, x1:x2]
                with cols[i % 5]:
                    st.image(final_cropped_pipe, caption=f"パイプ {i+1}", width=100)
        else:
            st.info("パイプは検出されませんでした。")
