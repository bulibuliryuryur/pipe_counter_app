import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import cv2

# クリック座標取得ライブラリ
from streamlit_image_coordinates import streamlit_image_coordinates

# Streamlit設定
st.set_page_config(page_title="パイプカウントアプリ (OpenVINO版)", layout="wide")
st.title("パイプの本数を数えるWebアプリ")

from openvino.runtime import Core
ie = Core()
ie.set_property({"CACHE_DIR": ""})

# --- モデルロード ---
def load_model():
    model_path = "last_openvino_model"
    model = YOLO(model_path)
    dummy = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
    _ = model.predict(dummy, conf=0.01, verbose=False)
    return model

model = load_model()

# --- 画像アップロード ---
uploaded_file = st.file_uploader("パイプの画像を選択してください", type=["jpg", "jpeg", "png"])

# 状態管理
if "detected" not in st.session_state:
    st.session_state.detected = False

if "points" not in st.session_state:
    st.session_state.points = []

# ★ 新しく追加：検出結果キャッシュ
if "auto_result" not in st.session_state:
    st.session_state.auto_result = None

if "auto_annotated" not in st.session_state:
    st.session_state.auto_annotated = None


if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="アップロード画像", use_column_width=True)

    conf_thres = st.slider("信頼度 (Confidence) の閾値", 0.0, 1.0, 0.5, 0.05)

    # 検出ボタン押下時
if st.button("パイプを検出"):
    st.session_state.detected = True
    st.session_state.auto_result = None
    st.session_state.auto_annotated = None
    st.session_state.points = []
    
    # クリック座標を保持する key をリセット
    st.session_state["click_image"] = None


    # 検出モードのとき
   # （上は前回コードと同じ）

if st.session_state.detected:

    # 検出は1回だけ
    if st.session_state.auto_result is None:
        results = model.predict(image, conf=conf_thres, verbose=False)
        result = results[0]
        st.session_state.auto_result = result
    else:
        result = st.session_state.auto_result

    num_pipes = len(result.boxes)

    # 青丸描画も1回だけ
    if st.session_state.auto_annotated is None:
        annotated_image = np.array(image)
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            cv2.circle(annotated_image, (cx, cy), 12, (0, 0, 255), -1)

        st.session_state.auto_annotated = annotated_image.copy()

    annotated_image = st.session_state.auto_annotated
    annotated_image_pil = Image.fromarray(annotated_image)

    st.subheader("自動検出結果 (青丸)")

    st.markdown("### ✍️ 手動カウント（赤丸を追加）")
    st.info("画像をクリックして赤い点を追加してください。")

    # ★ Undo ボタン
    if st.button("手動の点を削除"):
        if st.session_state.points:
            st.session_state.points = []
            st.session_state["click_image"] = None 
            

    # クリック座標取得
    click = streamlit_image_coordinates(
        annotated_image_pil,
        key="click_image",
    )

    # クリックされたら追加
    if click is not None and "x" in click and "y" in click:
        st.session_state.points.append((int(click["x"]), int(click["y"])))

    # 赤丸を描画
    manual_image = annotated_image.copy()
    for (px, py) in st.session_state.points:
        cv2.circle(manual_image, (px, py), 12, (255, 0, 0), -1)

    manual_pil = Image.fromarray(manual_image)

    st.image(manual_pil, caption="青＝自動 / 赤＝手動", use_column_width=True)

    manual_count = len(st.session_state.points)
    total_pipes = num_pipes + manual_count

    st.markdown("---")
    st.subheader("最終集計結果")

    st.success(f"合計パイプ本数： **{total_pipes} 本**")
    st.info(f"自動検出：{num_pipes} 本、手動追加：{manual_count} 本")
