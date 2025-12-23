import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from streamlit_cropper import st_cropper

# =========================
# Core: Otsu thresholding
# =========================
def otsu_threshold(gray_arr: np.ndarray) -> int:
    """Compute Otsu threshold for a uint8 grayscale array."""
    hist = np.bincount(gray_arr.ravel(), minlength=256).astype(np.float64)
    total = gray_arr.size
    if total == 0:
        return 128

    sum_total = np.dot(np.arange(256), hist)
    sumB = 0.0
    wB = 0.0
    maximum = -1.0
    threshold = 128

    for t in range(256):
        wB += hist[t]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break

        sumB += t * hist[t]
        mB = sumB / wB
        mF = (sum_total - sumB) / wF

        between = wB * wF * (mB - mF) ** 2
        if between > maximum:
            maximum = between
            threshold = t

    return int(threshold)

# =========================
# Core: dark/light ratio
# =========================
def dark_light_ratio_from_crop(crop_rgb: Image.Image, strip_white_thr: int = 245):
    """
    crop_rgb: PIL RGB image (already cropped to the strip area)
    strip_white_thr: pixels brighter than this are treated as background and ignored
    """
    gray = np.array(crop_rgb.convert("L")).astype(np.uint8)

    # Keep only non-white pixels to avoid counting tabletop/background.
    mask = gray < strip_white_thr
    valid = gray[mask]

    # If mask removes too much, fall back to using all pixels in crop
    if valid.size < 200:
        mask = np.ones_like(gray, dtype=bool)
        valid = gray.ravel()

    t = otsu_threshold(valid)

    dark = (gray <= t) & mask
    light = (gray > t) & mask

    dcnt = int(dark.sum())
    lcnt = int(light.sum())
    total = dcnt + lcnt

    dr = dcnt / total if total else 0.0
    lr = lcnt / total if total else 0.0

    return {
        "gray": gray,
        "mask": mask,
        "threshold": t,
        "dark": dark,
        "light": light,
        "dark_count": dcnt,
        "light_count": lcnt,
        "dark_ratio": dr,
        "light_ratio": lr,
        "valid_pixels": valid,
    }

# =========================
# Streamlit App
# =========================
st.set_page_config(page_title="試紙暗/淺比例分析（手動裁切）", layout="centered")
st.title("🧪試紙反應後局部褪色比例分析模型")
st.write("上傳圖片 → 用滑鼠拖曳裁切框只框住試紙 → 自動以 Otsu 閾值分成暗/淺 → 計算比例並視覺化。")

uploaded_file = st.file_uploader("請選擇一張圖片...", type=["jpg", "jpeg", "png", "bmp"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")

    st.subheader("1) 手動拖曳裁切框")
    st.caption("拖動四邊/角落調整範圍，讓框盡量只包含試紙本體（越乾淨越準）。")

    colA, colB, colC = st.columns([1, 1, 1])
    with colA:
        realtime_update = st.checkbox("拖曳時即時更新", value=True)
    with colB:
        box_color = st.color_picker("裁切框顏色", "#00FF00")
    with colC:
        aspect = st.selectbox("裁切框長寬比", ["不限制", "直立(1:4)", "橫放(5.5:1)"])

    aspect_ratio = None
    if aspect == "直立(1:4)":
        aspect_ratio = (1, 4)
    elif aspect == "橫放(5.5:1)":
        aspect_ratio = (5.5, 1)

    # Cropper returns a PIL image of the selected region
    col_crop, _ = st.columns([2, 2])  # 左邊較窄，右邊留白

    # 建立「顯示用」縮小影像（不影響原圖）
    display_img = img.copy()
    display_img.thumbnail((400, 400))  # 👈 控制正在裁切圖的最大長寬（可調）

    with col_crop:
      cropped_img = st_cropper(
        display_img,          # 👈 用縮小後的影像來裁切
        realtime_update=realtime_update,
        box_color=box_color,
        aspect_ratio=aspect_ratio,
        return_type="image",
      )

    st.image(cropped_img, caption="裁切後（分析範圍）", width=200)

    st.subheader("2) 分割設定")
    strip_white_thr = st.slider(
        "背景白色門檻（越高越嚴格剔除白色背景）",
        min_value=220,
        max_value=255,
        value=245,
        help="用來排除桌面/紙張等白色背景。若試紙很淡，可稍微降低。"
    )

    res = dark_light_ratio_from_crop(cropped_img, strip_white_thr=strip_white_thr)

    st.success(
        f"✅ 暗色比例：**{res['dark_ratio']:.2%}**  |  "
        f"淺色比例：**{res['light_ratio']:.2%}**  "
        f"（Otsu 閾值 = {res['threshold']}）"
    )

    st.subheader("3) 視覺化")

    # Histogram
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.hist(res["valid_pixels"].ravel(), bins=40, edgecolor="black")
    ax.axvline(res["threshold"], linestyle="--")
    ax.set_xlabel("Grayscale (0=black, 255=white)")
    ax.set_ylabel("Count")
    ax.set_title("Grayscale Histogram (counted pixels in crop)")
    st.pyplot(fig)

    # Overlay: dark=red, light=cyan (alpha blend)
    overlay = np.array(cropped_img).astype(np.float32)
    alpha = 0.35
    overlay[res["dark"]] = (1 - alpha) * overlay[res["dark"]] + alpha * np.array([255, 0, 0])       # dark
    overlay[res["light"]] = (1 - alpha) * overlay[res["light"]] + alpha * np.array([0, 255, 255])   # light
    overlay = overlay.clip(0, 255).astype(np.uint8)

    st.image(Image.fromarray(overlay), caption="分割疊圖：暗=紅、淺=青", width=200)

    st.subheader("4) 數值摘要")
    st.write(
        f"- Dark pixels: {res['dark_count']}\n"
        f"- Light pixels: {res['light_count']}\n"
        f"- Total counted: {res['dark_count'] + res['light_count']}"
    )
else:
    st.info("👆 請先上傳圖片開始分析。")




























