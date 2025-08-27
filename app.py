# app.py

import streamlit as st
from PIL import Image
import numpy as np
import cv2
import torch
from streamlit_cropper import st_cropper

st.set_page_config(page_title="パイプカウントアプリ", layout="wide")
st.title("パイプの本数を数えるWebアプリ")

# --- YOLOv5モデルのロード (Streamlitのキャッシュ機能を使って効率化) ---
@st.cache_resource
def load_yolo_model():
    """
    YOLOv5モデルをロードし、Streamlitのキャッシュに保存します。
    これにより、アプリの再起動時にモデルの再ロードが不要になります。
    """
    try:
        # PyTorch HubからYOLOv5モデルをロード
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model
    except Exception as e:
        st.error(f"モデルのロード中にエラーが発生しました。必要なライブラリがインストールされているか確認してください。\nエラー: {e}")
        return None

model = load_yolo_model()

# --- 画像ファイルアップロード機能 ---
st.subheader("画像をアップロードしてトリミング・カウント")
uploaded_file = st.file_uploader("パイプの画像を選択してください", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.info("画像をアップロードしてアプリを使い始めてください。")
else:
    original_image = Image.open(uploaded_file)
    st.image(original_image, caption="元の画像", use_column_width=True)

    st.write("---")
    st.subheader("画像をトリミングしてください")
    st.info("画像上でドラッグしてトリミング範囲を選択できます。枠の四隅や辺を動かして自由に調整してください。")

    # st_cropper を使って画像をインタラクティブにトリミング
    # アスペクト比を固定せずに自由なトリミングを許可
    cropped_image = st_cropper(
        img_file=original_image,
        box_color="#0000FF", # トリミング枠の色を青に設定
        return_type="image", # PIL Imageオブジェクトとしてトリミング結果を返す
        aspect_ratio=None # アスペクト比を固定しない
    )
    
    if cropped_image:
        st.write("---")
        st.subheader("トリミング後の画像")
        st.image(cropped_image, caption="トリミングされた画像", use_column_width=True)

        if st.button("トリミングして解析"):
            with st.spinner("画像を解析しています..."):
                try:
                    # PIL ImageをNumPy配列に変換
                    img_np = np.array(cropped_image)

                    # YOLOv5で推論を実行
                    # YOLOv5はPIL ImageやNumPy配列を直接扱えるため、変換が簡潔
                    results = model(img_np, size=640)
                    
                    # 検出結果の描画
                    # YOLOv5の.render()メソッドで検出結果を描画
                    annotated_frame = np.squeeze(results.render())
                    # OpenCV形式（BGR）に変換してStreamlitに表示
                    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                    
                    st.image(annotated_frame, caption="検出結果", use_column_width=True, channels="BGR")

                    # 検出されたオブジェクトの数をカウント
                    predictions = results.pred[0]
                    num_pipes = len(predictions)

                    # Markdownで見出しとして大きく表示
                    st.markdown(f"### 検出されたパイプの数: <span style='color:green;'>**{num_pipes}本**</span>", unsafe_allow_html=True)
                    
                    # 個別のパイプをトリミングして表示
                    st.subheader("個別のパイプ画像 (検出後)")
                    if num_pipes > 0:
                        cols = st.columns(min(num_pipes, 5))
                        for i, box in enumerate(predictions):
                            x1, y1, x2, y2 = map(int, box[:4])
                            # 元のcropped_imageからさらにトリミング
                            final_cropped_pipe = img_np[y1:y2, x1:x2]
                            with cols[i % 5]:
                                st.image(final_cropped_pipe, caption=f"パイプ {i+1}", width=100)
                    else:
                        st.info("トリミングされた画像中にパイプは検出されませんでした。")
                
                except Exception as e:
                    st.error(f"解析中にエラーが発生しました: {e}")