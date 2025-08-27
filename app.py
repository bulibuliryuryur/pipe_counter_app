import streamlit as st
from PIL import Image
import numpy as np
import cv2
import torch

from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="パイプカウントアプリ(YOLOv5)", layout="wide")
st.title("パイプの本数を数えるWebアプリ(YOLOv5版)")

# --- YOLOv5モデルのロード ---
@st.cache_resource
def load_yolo_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
    return model

model = load_yolo_model()

# --- 画像ファイルアップロード機能 ---
st.subheader("画像をアップロードしてトリミング・カウント")
uploaded_file = st.file_uploader("パイプの画像を選択してください", type=["jpg", "jpeg", "png"])

original_image = None
if uploaded_file is not None:
    original_image = Image.open(uploaded_file)
    st.image(original_image, caption="元の画像", use_column_width=True)

    st.write("---")
    st.subheader("画像をトリミングしてください")
    st.info("画像上でドラッグしてトリミング範囲を選択し、'トリミングして解析' ボタンをクリックしてください。")

    value = streamlit_image_coordinates(original_image, key=f"image_coordinates_{uploaded_file.name}")

    cropped_image = None
    if value:
        x_min = int(value['x'])
        y_min = int(value['y'])
        width = int(value['width'])
        height = int(value['height'])

        cropped_image = original_image.crop((x_min, y_min, x_min + width, y_min + height))
        
        st.write("---")
        st.subheader("トリミング後の画像")
        st.image(cropped_image, caption="トリミングされた画像", use_column_width=True)

    if st.button("トリミングして解析", disabled=(cropped_image is None)):
        if cropped_image:
            img_np = np.array(cropped_image)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            with st.spinner("画像を解析しています..."):
                results = model(img_bgr, conf=0.5)

            annotated_frame = results.render()[0] 
            st.image(annotated_frame, caption="検出結果", use_column_width=True, channels="BGR")

            detections = results.pred[0]
            num_pipes = len(detections)
            st.success(f"検出されたパイプの数: **{num_pipes}本**")

            st.subheader("個別のパイプ画像 (検出後)")
            if num_pipes > 0:
                cols = st.columns(min(num_pipes, 5))
                for i, box in enumerate(detections):
                    x1, y1, x2, y2 = map(int, box[:4])
                    final_cropped_pipe = img_bgr[y1:y2, x1:x2]
                    with cols[i % 5]:
                        st.image(final_cropped_pipe, caption=f"パイプ {i+1}", width=100, channels="BGR")
            else:
                st.info("トリミングされた画像中にパイプは検出されませんでした。")
        else:
            st.warning("画像をトリミングしてから 'トリミングして解析' ボタンをクリックしてください。")