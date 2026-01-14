import streamlit as st
from PIL import Image, ImageDraw
from ultralytics import YOLO
import numpy as np
import cv2
from streamlit_image_coordinates import streamlit_image_coordinates
import io
import zipfile
from datetime import datetime

# -------------------------------------------------
# 1. 初期設定とモデルロード
# -------------------------------------------------
st.set_page_config(page_title="Pipe Counter Pro", layout="wide")

@st.cache_resource
def load_model():
    model_path = "last_openvino_model" # フォルダ名を指定
    model = YOLO(model_path)
    return model

model = load_model()
TARGET_SIZE = 960

# -------------------------------------------------
# 2. Session State の初期化 (状態管理)
# -------------------------------------------------
if "phase" not in st.session_state: st.session_state.phase = "upload"
if "mask_rects" not in st.session_state: st.session_state.mask_rects = []
if "temp_point" not in st.session_state: st.session_state.temp_point = None
if "last_processed_click" not in st.session_state: st.session_state.last_processed_click = None
if "points" not in st.session_state: st.session_state.points = [] 
if "auto_boxes" not in st.session_state: st.session_state.auto_boxes = None
if "manual_minus" not in st.session_state: st.session_state.manual_minus = 0

# -------------------------------------------------
# 3. 便利関数
# -------------------------------------------------
def apply_masks(img, rects):
    """画像に黒塗りのマスクを適用する"""
    draw = ImageDraw.Draw(img)
    for r in rects:
        draw.rectangle([r[0], r[1], r[2], r[3]], fill="black")
    return img

def resize_with_aspect_ratio(img, target_max_side):
    """アスペクト比を維持してリサイズする"""
    w, h = img.size
    if w > h:
        new_w = target_max_side
        new_h = int(h * (target_max_side / w))
    else:
        new_h = target_max_side
        new_w = int(w * (target_max_side / h))
    return img.resize((new_w, new_h), Image.LANCZOS)

# -------------------------------------------------
# 4. メイン UI フロー
# -------------------------------------------------
st.title("PPAP(Pipe Perceive APp)")

uploaded_file = st.file_uploader("パイプの画像を選択してください", type=["jpg", "jpeg", "png"])

if uploaded_file:
    original_image = Image.open(uploaded_file).convert("RGB")
    # --- アスペクト比を維持してリサイズ ---
    base_image = resize_with_aspect_ratio(original_image, TARGET_SIZE)
    display_width, display_height = base_image.size

    # --- PHASE: UPLOAD (初期画面) ---
    if st.session_state.phase == "upload":
        st.image(base_image, caption="アップロード画像", width=display_width)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✂️ 検出エリアを編集（除外設定）", use_container_width=True):
                st.session_state.phase = "edit"
                st.rerun()
        with col2:
            if st.button("🚀 そのままパイプを検出", type="primary", use_container_width=True):
                st.session_state.phase = "detect"
                st.rerun()

    # --- PHASE: EDIT (除外設定画面) ---
    elif st.session_state.phase == "edit":
        st.subheader("除外エリアの設定 (2点クリックで四角形を作成)")
        
        edit_img = base_image.copy()
        edit_img = apply_masks(edit_img, st.session_state.mask_rects)
        
        if st.session_state.temp_point:
            draw = ImageDraw.Draw(edit_img)
            tx, ty = st.session_state.temp_point
            draw.ellipse([tx-8, ty-8, tx+8, ty+8], fill="red")

        # 表示サイズに合わせて座標を取得
        coords = streamlit_image_coordinates(edit_img, key="mask_editor", width=display_width)

        if coords:
            curr_c = (coords["x"], coords["y"])
            if curr_c != st.session_state.last_processed_click:
                st.session_state.last_processed_click = curr_c
                if st.session_state.temp_point is None:
                    st.session_state.temp_point = curr_c
                else:
                    p1 = st.session_state.temp_point
                    p2 = curr_c
                    rect = (min(p1[0], p2[0]), min(p1[1], p2[1]), max(p1[0], p2[0]), max(p1[1], p2[1]))
                    st.session_state.mask_rects.append(rect)
                    st.session_state.temp_point = None
                st.rerun()

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("✅ 編集を確定して検出へ", type="primary"):
                st.session_state.phase = "detect"
                st.rerun()
        with col2:
            if st.button("↩️ 1つ戻す"):
                if st.session_state.mask_rects: st.session_state.mask_rects.pop()
                st.rerun()
        with col3:
            if st.button("❌ 全リセットして戻る"):
                st.session_state.mask_rects = []
                st.session_state.phase = "upload"
                st.rerun()

    # --- PHASE: DETECT (検出・手動補正・保存画面) ---
    elif st.session_state.phase == "detect":
        st.subheader("検出結果の確認と補正")
        
        inference_img = base_image.copy()
        inference_img = apply_masks(inference_img, st.session_state.mask_rects)
        
        if st.session_state.auto_boxes is None:
            with st.spinner("AIがパイプを検出中..."):
                # YOLOは内部でパディングして処理するため、アスペクト比維持した画像のままでOK
                results = model.predict(inference_img, conf=0.5, max_det=2000, verbose=False)
                boxes = []
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                    boxes.append((x1, y1, x2, y2))
                st.session_state.auto_boxes = boxes

        display_np = np.array(inference_img)
        for (x1, y1, x2, y2) in st.session_state.auto_boxes:
            cx, cy = int((x1+x2)/2), int((y1+y2)/2)
            cv2.circle(display_np, (cx, cy), 13, (255, 255, 0), -1)
        for (px, py) in st.session_state.points:
            cv2.circle(display_np, (px, py), 13, (255, 0, 0), -1)

        # --- カウント調整用 UI ---
        num_auto = len(st.session_state.auto_boxes)
        num_manual = len(st.session_state.points)
        num_minus = st.session_state.manual_minus
        total_pipes = max(0, num_auto + num_manual - num_minus)

        st.success(f"合計： {total_pipes} 本 (自動: {num_auto} / 手動追加: {num_manual} / 削減分: -{num_minus})")

        adj_col1, adj_col2, adj_col3 = st.columns(3)
        with adj_col1:
            if st.button("➕ 手動追加を1つ消す", use_container_width=True):
                if st.session_state.points:
                    st.session_state.points.pop()
                    st.rerun()
        with adj_col2:
            if st.button("➖ カウントを1つ減らす", use_container_width=True):
                st.session_state.manual_minus += 1
                st.rerun()
        with adj_col3:
            if st.button("🔄 減らした分をリセット", use_container_width=True):
                st.session_state.manual_minus = 0
                st.rerun()

        st.info("画像をクリックしてカウントを手動で追加できます。")
        # 表示サイズに合わせて座標を取得
        fix_coords = streamlit_image_coordinates(Image.fromarray(display_np), key="manual_fix", width=display_width)

        if fix_coords:
            new_p = (fix_coords["x"], fix_coords["y"])
            if new_p != st.session_state.last_processed_click:
                st.session_state.last_processed_click = new_p
                st.session_state.points.append(new_p)
                st.rerun()

        # --- 保存エリア ---
        st.markdown("---")
        st.subheader("💾 結果を保存")
        
        col_meta1, col_meta2 = st.columns(2)
        with col_meta1:
            group = st.selectbox("グループ", ["A", "B", "C", "D"])
        with col_meta2:
            comment = st.text_area("コメント（任意）")

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as z:
            img_buf = io.BytesIO()
            Image.fromarray(display_np).save(img_buf, format="PNG")
            z.writestr(f"result_{timestamp}.png", img_buf.getvalue())

            csv_text = "日時,グループ,自動検出本数,手動追加本数,手動削減本数,合計本数,コメント\n"
            csv_text += f"{timestamp.replace('_',' ')},{group},{num_auto}本,{num_manual}本,{num_minus}本,{total_pipes}本,\"{comment}\""
            z.writestr(f"result_{timestamp}.csv", csv_text.encode("utf-8-sig"))

        st.download_button(
            label="⬇ 結果をZIPでダウンロード",
            data=zip_buffer.getvalue(),
            file_name=f"pipe_result_{timestamp}.zip",
            mime="application/zip",
            type="primary"
        )

        st.markdown("---")
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("⏮ エリア編集に戻る"):
                st.session_state.auto_boxes = None 
                st.session_state.phase = "edit"
                st.rerun()
        with col_btn2:
            if st.button("🗑 手動点と削減分を全リセット"):
                st.session_state.points = []
                st.session_state.manual_minus = 0
                st.rerun()