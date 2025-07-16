# app.py

import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
# from streamlit_webrtc import webrtc_streamer, VideoTransformerBase # 今回は使用しないのでコメントアウト
from streamlit_image_coordinates import streamlit_image_coordinates # 新しくインポート

st.set_page_config(page_title="パイプカウントアプリ", layout="wide")
st.title("パイプの本数を数えるWebアプリ")

# --- YOLOv8モデルのロード (Streamlitのキャッシュ機能を使って効率化) ---
@st.cache_resource
def load_yolo_model():
    model = YOLO('yolov8n.pt') # 事前学習済みモデルを使用。必要に応じてカスタムモデルパスに。
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

    # streamlit_image_coordinates を使ってトリミング範囲を選択
    # 'key' を設定して、アップロードごとにウィジェットがリフレッシュされるようにする
    value = streamlit_image_coordinates(original_image, key=f"image_coordinates_{uploaded_file.name}")

    cropped_image = None
    if value:
        # 選択された座標を取得
        # value は辞書で、'x', 'y', 'width', 'height' が含まれる
        x_min = int(value['x'])
        y_min = int(value['y'])
        width = int(value['width'])
        height = int(value['height'])

        # PIL Image をトリミング
        # Image.crop((left, upper, right, lower))
        cropped_image = original_image.crop((x_min, y_min, x_min + width, y_min + height))
        
        st.write("---")
        st.subheader("トリミング後の画像")
        st.image(cropped_image, caption="トリミングされた画像", use_column_width=True)

    if st.button("トリミングして解析", disabled=(cropped_image is None)): # トリミング画像がなければボタンを無効化
        if cropped_image:
            # PIL ImageをOpenCV形式 (BGR) に変換
            img_np = np.array(cropped_image)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            st.write("画像を解析しています...")
            # YOLOv8で推論を実行
            results = model(img_bgr, conf=0.5, verbose=False) # confは信頼度閾値、verbose=Falseでログを抑制

            # 検出結果の描画
            annotated_frame = results[0].plot() # YOLOv8のplot機能で検出結果を描画
            st.image(annotated_frame, caption="検出結果", use_column_width=True, channels="BGR")

            # 検出されたオブジェクト（パイプ）の数をカウント
            num_pipes = len(results[0].boxes)
            st.success(f"検出されたパイプの数: **{num_pipes}本**")

            # 必要であれば、個別のパイプをトリミングして表示 (今回はトリミング済み画像からさらに検出なので、見送りも可)
            st.subheader("個別のパイプ画像 (検出後)")
            if num_pipes > 0:
                cols = st.columns(min(num_pipes, 5)) # 最大5列で表示
                for i, box in enumerate(results[0].boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # 元のcropped_imageからさらにトリミング
                    final_cropped_pipe = img_bgr[y1:y2, x1:x2]
                    with cols[i % 5]: # 5列ごとに改行
                        st.image(final_cropped_pipe, caption=f"パイプ {i+1}", width=100, channels="BGR")
            else:
                st.info("トリミングされた画像中にパイプは検出されませんでした。")
        else:
            st.warning("画像をトリミングしてから 'トリミングして解析' ボタンをクリックしてください。")

# --- Webカメラ撮影機能 (今回はコメントアウトのまま) ---
# st.markdown("---")
# st.subheader("Webカメラで撮影してカウント (実験的機能)")
# st.info("Webカメラ機能はまだ調整中です。まずは画像アップロードをお試しください。")