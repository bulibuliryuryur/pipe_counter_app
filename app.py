import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import torch

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
    img_width, img_height = original_image.size

    st.write("---")
    st.subheader("トリミング範囲の指定")
    st.info("スライダーを動かすと、下の画像にプレビューが表示されます。")

    # セッション状態に座標を保存（スライダーの値を保持するため）
    if 'x1' not in st.session_state:
        st.session_state.x1 = 0
    if 'x2' not in st.session_state:
        st.session_state.x2 = img_width
    if 'y1' not in st.session_state:
        st.session_state.y1 = 0
    if 'y2' not in st.session_state:
        st.session_state.y2 = img_height

    # スライダーと数値入力でトリミング範囲を設定
    x1, x2 = st.slider("横方向の範囲 (X軸)", 0, img_width, (st.session_state.x1, st.session_state.x2))
    y1, y2 = st.slider("縦方向の範囲 (Y軸)", 0, img_height, (st.session_state.y1, st.session_state.y2))
    
    # スライダーの値をセッション状態に更新
    st.session_state.x1 = x1
    st.session_state.x2 = x2
    st.session_state.y1 = y1
    st.session_state.y2 = y2

    # --- プレビュー画像の生成 ---
    # 元の画像をコピーして、その上に矩形を描画
    preview_image = original_image.copy()
    draw = ImageDraw.Draw(preview_image)
    
    # 矩形を描画 (線の太さと色を指定)
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    
    # プレビューを表示
    st.image(preview_image, caption="トリミング範囲プレビュー", use_column_width=True)

    # --- トリミングされた画像の解析 ---
    if st.button("この範囲でパイプを解析"):
        # プレビュー表示に使った座標で画像をトリミング
        cropped_image_pil = original_image.crop((x1, y1, x2, y2))
        cropped_image = np.array(cropped_image_pil)
        
        st.write("---")
        st.subheader("解析結果")
        st.image(cropped_image, caption="トリミングされた画像", use_column_width=True)

        img_rgb = cropped_image.copy()
        with st.spinner("解析中..."):
            results = model(img_rgb)
        
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