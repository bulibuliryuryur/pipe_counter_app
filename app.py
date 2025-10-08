import streamlit as st
from PIL import Image
import torch
from ultralytics import YOLO # YOLOv8のインポート

# Streamlitの設定
st.set_page_config(page_title="パイプカウントアプリ(YOLOv8版)", layout="wide")
st.title("パイプの本数を数えるWebアプリ (YOLOv8版)")

# --- モデルロード ---
# YOLOv8では、デバイスの指定もYOLOクラスが自動で処理します
@st.cache_resource
def load_model():
    # last.ptのパスを正確に指定してください。
    # ここでは、Streamlitアプリと同じディレクトリにあると仮定します。
    # 絶対パスが必要な場合は、適宜変更してください。
    model_path = "last.pt" 
    
    # YOLOv8モデルのロード
    try:
        model = YOLO(model_path)
        return model
    except FileNotFoundError:
        st.error(f"モデルファイルが見つかりません: {model_path}")
        st.stop()
    except Exception as e:
        st.error(f"モデルのロード中にエラーが発生しました: {e}")
        st.stop()

try:
    model = load_model()
    # モデルがロードされたことを確認するためのオプションのメッセージ
    # st.success("YOLOv8モデルを正常にロードしました。")
except Exception:
    # load_model内でエラー処理を行っているため、ここではpass
    pass

# --- 画像アップロード ---
uploaded_file = st.file_uploader("パイプの画像を選択してください", type=["jpg", "jpeg", "png"])
if uploaded_file:
    # PIL Imageとしてロード (YOLOv8のmodel()メソッドが直接扱えます)
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="アップロード画像", use_column_width=True)
    
    # 信頼度閾値の設定
    conf_thres = st.slider("信頼度 (Confidence) の閾値", 0.0, 1.0, 0.5, 0.05)
    
    if st.button("パイプを検出"):
        with st.spinner('検出中...'):
            # --- 推論実行 (YOLOv8) ---
            # source=imageで直接PIL Imageを渡します。
            # conf=conf_thresで信頼度閾値を設定します。
            # verbose=Falseでコンソールへの詳細出力を抑止します。
            results = model.predict(source=image, conf=conf_thres, verbose=False)

            # --- 本数カウントと描画 ---
            num_pipes = 0
            
            # 1枚の画像に対する結果を取得
            if results and len(results) > 0:
                result = results[0]
                
                # 検出されたオブジェクトの数をカウント
                num_pipes = len(result.boxes)
                
                # 検出結果が描画されたPIL Imageを取得
                # 'plot()'メソッドでバウンディングボックスなどが描画されます
                annotated_image = result.plot(
                    conf=True,     # 信頼度を表示
                    line_width=2,  # 線の太さ
                    # 必要に応じてその他の引数を追加 (例: labels=True, boxes=True)
                )
                
                # numpy配列からPIL Imageに変換し直して表示
                if isinstance(annotated_image, torch.Tensor):
                    annotated_image_np = annotated_image.permute(1, 2, 0).cpu().numpy()
                else:
                    annotated_image_np = annotated_image
                
                annotated_image_pil = Image.fromarray(annotated_image_np)
                
                st.subheader("検出結果")
                st.image(annotated_image_pil, caption=f"検出されたパイプ: {num_pipes}本", use_column_width=True)

            # --- 結果表示 ---
            if num_pipes > 0:
                st.success(f"検出されたパイプの数: **{num_pipes}本**")
            else:
                st.info("パイプは検出されませんでした。閾値を変えて再試行してください。")