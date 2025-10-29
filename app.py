import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np

# Streamlitの設定
st.set_page_config(page_title="パイプカウントアプリ (OpenVINO版)", layout="wide")
st.title("パイプの本数を数えるWebアプリ (OpenVINO版)")

# --- モデルロード(OpenVINO) ---
@st.cache_resource
def load_model():
    model_path = "last_openvino_model"  # フォルダを指定
    
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"モデルのロード中にエラーが発生しました: {e}")
        st.stop()

model = load_model()

# --- 画像アップロード ---
uploaded_file = st.file_uploader("パイプの画像を選択してください", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="アップロード画像", use_column_width=True)

    conf_thres = st.slider("信頼度 (Confidence) の閾値", 0.0, 1.0, 0.5, 0.05)

    if st.button("パイプを検出"):
        with st.spinner('検出中...'):
            
            results = model.predict(image, conf=conf_thres, verbose=False)
            result = results[0]

            num_pipes = len(result.boxes)

            # 描画画像取得
            annotated_image = result.plot(line_width=2, conf=True)
            
            # numpy 配列を PIL に変換
            if isinstance(annotated_image, np.ndarray):
                annotated_image_pil = Image.fromarray(annotated_image)
            else:
                annotated_image_pil = Image.fromarray(np.array(annotated_image))

            st.subheader("検出結果")
            st.image(
                annotated_image_pil, 
                caption=f"検出されたパイプ: {num_pipes}本",
                use_column_width=True
            )

            if num_pipes > 0:
                st.success(f"検出されたパイプの数: **{num_pipes}本**")
            else:
                st.info("パイプは検出されませんでした。閾値を変更して再試行してください。")
