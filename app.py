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
    # ultralytics/yolov5 から yolov5n モデルを取得
    model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
    return model

model = load_yolo_model()

# --- 画像ファイルアップロード機能 ---
st.subheader("画像をアップロードしてトリミング・カウント")
uploaded_file = st.file_uploader("パイプの画像を選択してください", type=["jpg", "jpeg", "png"])

if uploaded_file:
    original_image = Image.open(uploaded_file)
    st.image(original_image, caption="元の画像", use_column_width=True)

    st.write("---")
    st.subheader("画像をトリミングしてください")
    st.info("赤枠をドラッグして範囲を選択してください。")

    # --- streamlit-cropper でトリミング ---
    cropped_image = st_cropper(original_image, realtime_update=True, box_color='#FF0000', aspect_ratio=None)

    st.write("---")
    st.subheader("トリミング後の画像")
    st.image(cropped_image, caption="トリミングされた画像", use_column_width=True)

    # --- 解析ボタン ---
    if st.button("トリミングして解析"):
        img_np = np.array(cropped_image)   # PIL → numpy
        img_rgb = img_np.copy()            # RGB のまま渡す

        with st.spinner("画像を解析しています..."):
            results = model(img_rgb)

        annotated_frame = results.render()[0]
        st.image(annotated_frame, caption="検出結果", use_column_width=True, channels="RGB")

        detections = results.pred[0]
        num_pipes = len(detections)
        st.success(f"検出されたパイプの数: **{num_pipes}本**")

        st.subheader("個別のパイプ画像 (検出後)")
        if num_pipes > 0:
            cols = st.columns(min(num_pipes, 5))
            for i, box in enumerate(detections):
                x1, y1, x2, y2 = map(int, box[:4])
                final_cropped_pipe = img_rgb[y1:y2, x1:x2]
                with cols[i % 5]:
                    st.image(final_cropped_pipe, caption=f"パイプ {i+1}", width=100)
        else:
            st.info("トリミングされた画像中にパイプは検出されませんでした。")
