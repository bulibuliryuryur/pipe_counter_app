import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import cv2

from streamlit_image_coordinates import streamlit_image_coordinates
from openvino.runtime import Core

import json
from datetime import datetime
import io
import zipfile

# -------------------------------------------------
# 定数
# -------------------------------------------------
TARGET_SIZE = 960

# -------------------------------------------------
# Streamlit 設定
# -------------------------------------------------
st.set_page_config(page_title="パイプカウントアプリ (OpenVINO版)", layout="wide")
st.title("パイプの本数を数えるWebアプリ")

ie = Core()
ie.set_property({"CACHE_DIR": ""})

# -------------------------------------------------
# モデルロード
# -------------------------------------------------
def load_model():
    model_path = "last_openvino_model"
    model = YOLO(model_path)
    dummy = np.random.randint(0, 256, (TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8)
    _ = model.predict(dummy, conf=0.01, verbose=False)
    return model

model = load_model()

# -------------------------------------------------
# 状態管理
# -------------------------------------------------
if "detected" not in st.session_state:
    st.session_state.detected = False
if "points" not in st.session_state:
    st.session_state.points = []
if "pending_point" not in st.session_state:
    st.session_state.pending_point = None
if "last_click_id" not in st.session_state:
    st.session_state.last_click_id = None
if "auto_boxes" not in st.session_state:
    st.session_state.auto_boxes = None
if "auto_annotated" not in st.session_state:
    st.session_state.auto_annotated = None

# -------------------------------------------------
# 画像アップロード
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "パイプの画像を選択してください",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    original_image = Image.open(uploaded_file).convert("RGB")
    base_image = original_image.resize((TARGET_SIZE, TARGET_SIZE))
    st.image(base_image, caption="アップロード画像（960×960）", use_column_width=True)

    conf_thres = st.slider("信頼度 (Confidence)", 0.0, 1.0, 0.5, 0.05)

# -------------------------------------------------
# 検出ボタン
# -------------------------------------------------
if st.button("パイプを検出"):
    st.session_state.detected = True
    st.session_state.points = []
    st.session_state.pending_point = None
    st.session_state.last_click_id = None
    st.session_state.auto_boxes = None
    st.session_state.auto_annotated = None
    st.session_state["click_image"] = None

# -------------------------------------------------
# 検出処理
# -------------------------------------------------
if st.session_state.detected and uploaded_file:

    base_image = original_image.resize((TARGET_SIZE, TARGET_SIZE))
    base_np = np.array(base_image)

    # YOLO 推論
    if st.session_state.auto_boxes is None:
        results = model.predict(
            base_image,
            conf=conf_thres,
            max_det=2000,
            verbose=False
        )
        boxes = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            boxes.append((x1, y1, x2, y2))
        st.session_state.auto_boxes = boxes
    else:
        boxes = st.session_state.auto_boxes

    num_pipes = len(boxes)

    # 自動検出描画
    if st.session_state.auto_annotated is None:
        annotated = base_np.copy()
        for x1, y1, x2, y2 in boxes:
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            cv2.circle(annotated, (cx, cy), 15, (255, 255, 0), -1)
        st.session_state.auto_annotated = annotated.copy()

    annotated_image = st.session_state.auto_annotated

    st.subheader("自動検出結果（黄色）")
    st.markdown("### ✍️ 手動カウント（赤）")

    display_width = st.slider("表示サイズ", 400, 1200, 900)
    display_width = min(display_width, TARGET_SIZE)
    ratio = TARGET_SIZE / display_width

    base_img = annotated_image.copy()

    for px, py in st.session_state.points:
        cv2.circle(base_img, (px, py), 15, (255, 0, 0), -1)

    if st.session_state.pending_point:
        px, py = st.session_state.pending_point
        cv2.circle(base_img, (px, py), 15, (255, 0, 0), -1)

    click = streamlit_image_coordinates(
        Image.fromarray(base_img),
        key="click_image",
        width=display_width
    )

    if click and "x" in click and "y" in click:
        cid = (click["x"], click["y"])
        if cid != st.session_state.last_click_id:
            st.session_state.pending_point = (
                int(click["x"] * ratio),
                int(click["y"] * ratio)
            )
            st.session_state.last_click_id = cid

    col1, col2 = st.columns(2)
    with col1:
        if st.button("更新（確定）"):
            if st.session_state.pending_point:
                st.session_state.points.append(st.session_state.pending_point)
                st.session_state.pending_point = None
    with col2:
        if st.button("手動点を全削除"):
            st.session_state.points = []
            st.session_state.pending_point = None
            st.session_state.last_click_id = None

    manual_count = len(st.session_state.points)
    total_pipes = num_pipes + manual_count

    st.markdown("---")
    st.success(f"合計パイプ本数： {total_pipes} 本")
    st.info(f"自動：{num_pipes} ／ 手動：{manual_count}")

    # -------------------------------------------------
    # ZIP保存（ダウンロード）
    # -------------------------------------------------
    st.markdown("---")
    st.subheader("💾 結果を保存（ZIP）")

    group = st.selectbox("グループ", ["A", "B", "C", "D"])
    comment = st.text_area("コメント（任意）")

    if st.button("📦 ZIPをダウンロード"):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as z:

            # 画像
            img_buf = io.BytesIO()
            Image.fromarray(base_img).save(img_buf, format="PNG")
            z.writestr("result.png", img_buf.getvalue())

            # JSON
            meta = {
                "datetime": timestamp,
                "group": group,
                "auto_count": num_pipes,
                "manual_count": manual_count,
                "total_count": total_pipes
            }
            z.writestr("data.json", json.dumps(meta, ensure_ascii=False, indent=2))

            # コメント
            z.writestr("comment.txt", comment)

        st.download_button(
            label="⬇ ZIPダウンロード",
            data=zip_buffer.getvalue(),
            file_name=f"pipe_result_{timestamp}.zip",
            mime="application/zip"
        )
