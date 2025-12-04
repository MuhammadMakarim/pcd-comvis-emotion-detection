from pathlib import Path
from datetime import datetime
import tempfile

import av
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from cv2 import data as cv2_data
from streamlit_webrtc import (
    RTCConfiguration,
    VideoTransformerBase,
    WebRtcMode,
    webrtc_streamer,
)

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import timm
from PIL import Image
import gdown
from typing import cast
from torch import Tensor


# ----------------------------
# KONFIGURASI GLOBAL
# ----------------------------

CLASS_LABELS = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', 'Surprise']
POSITIVE_EMOTIONS = {'Happy', 'Surprise'}

IM_SIZE = 224
device = "cuda" if torch.cuda.is_available() else "cpu"

# === MODEL .PTH DARI LINK (SUDAH DIISI DENGAN LINK ANDA) ===
MODEL_URL = "https://drive.google.com/uc?id=179lNj9jBBFaurPupuhCn15y_359TEjrL"
MODEL_PATH = Path("emotion_model.pth")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Haarcascade untuk face detection (sesuai pipeline training)
face_cascade = cv2.CascadeClassifier(
    cv2_data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Transform sama dengan notebook PyTorch
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

transform = T.Compose([
    T.Resize((IM_SIZE, IM_SIZE)),
    T.Grayscale(num_output_channels=3),
    T.ToTensor(),
    T.Normalize(mean=mean, std=std),
])

# Artikel penjelasan emosi vs kinerja (boleh Anda edit)
ARTICLE_TEXT = """
Berbagai penelitian di bidang manajemen dan psikologi organisasi menunjukkan bahwa
**emosi karyawan sangat berpengaruh terhadap kinerja** dan kualitas output pekerjaan.

1. **Emosi positif (misalnya senang, antusias, dan bangga)**:
   - Berkaitan dengan **kinerja yang lebih baik**, pelayanan yang lebih ramah, dan kreativitas yang lebih tinggi.
   - Karyawan cenderung lebih termotivasi, kooperatif, dan mau membantu rekan kerja.
   - Studi berbasis experience sampling menemukan bahwa ketika karyawan berada dalam
     emosi positif, penilaian terhadap kualitas layanan dan produktivitas harian cenderung meningkat.

2. **Emosi negatif (misalnya marah, sedih, takut, cemas)**:
   - Dapat menurunkan fokus, meningkatkan konflik, dan menurunkan kualitas pengambilan keputusan.
   - Dalam jangka panjang, dominasi emosi negatif dapat berkaitan dengan **burnout, penurunan
     kepuasan kerja, dan niat resign**.
   - Emosi seperti marah dan frustrasi sering membuat karyawan lebih egosentris, sehingga
     kemampuan melihat perspektif orang lain menurun. Hal ini berdampak pada kerja tim dan
     kualitas pelayanan ke pelanggan.

3. **Variasi emosi sehari-hari juga penting**:
   - Fluktuasi emosi yang terlalu besar, terutama jika didominasi emosi negatif, berhubungan
     dengan kelelahan emosional dan turunnya kepuasan kerja.
   - Sebaliknya, ketika organisasi mampu menyediakan lingkungan dan sumber daya kerja yang baik,
     karyawan lebih banyak merasakan emosi positif sehingga kinerja hariannya meningkat.

4. **Peran organisasi dan atasan**:
   - Kepemimpinan yang memperhatikan sisi emosional karyawan (emotional leadership) dapat
     meningkatkan emosi positif bawahan dan pada akhirnya meningkatkan kinerja mereka.
   - Pelatihan **emotional intelligence** membantu karyawan mengenali, mengelola, dan
     mengekspresikan emosi dengan lebih sehat, sehingga dampak negatif ke kinerja bisa dikurangi.

Secara praktis, pemantauan emosi bukan untuk “menghakimi” karyawan, tetapi:
- Sebagai **alat deteksi dini** apabila ada kecenderungan emosi negatif yang berlarut-larut
  (misalnya sering sedih atau marah).
- Menjadi dasar bagi manajer dan HR untuk memberikan dukungan, konseling, atau perbaikan
  lingkungan kerja agar karyawan dapat kembali ke kondisi emosi yang lebih sehat dan produktif.
"""

# ----------------------------
# LOAD MODEL .PTH DARI LINK
# ----------------------------

@st.cache_resource(show_spinner=True)
def load_model():
    if not MODEL_PATH.exists():
        try:
            st.info("Mengunduh model (.pth) dari Google Drive, harap tunggu...")
            gdown.download(MODEL_URL, str(MODEL_PATH), quiet=False)
        except Exception as exc:
            st.error(f"Gagal mengunduh model dari link: {exc}")
            raise

    model = timm.create_model(
        "densenet169",
        pretrained=False,
        num_classes=len(CLASS_LABELS)
    )
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# ----------------------------
# STATE & LOGGING
# ----------------------------

def ensure_history_state():
    if "history" not in st.session_state:
        st.session_state.history = []


def log_prediction(source: str, employee_name: str, shift: str, label: str, confidence: float):
    ensure_history_state()
    st.session_state.history.append(
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": source,
            "employee": employee_name or "Tanpa Nama",
            "shift": shift,
            "label": label,
            "confidence": round(confidence, 4),
        }
    )


# ----------------------------
# PREPROCESS & PREDICT (BGR FRAME)
# ----------------------------

def preprocess_and_detect_face(bgr_image: np.ndarray, return_bbox: bool = False):
    """
    Pipeline:
    - BGR -> Grayscale
    - CLAHE
    - Haarcascade face detection
    - ROI wajah (kalau tidak ada wajah -> seluruh frame)
    """
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    faces = face_cascade.detectMultiScale(gray_eq, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        face_roi = gray_eq
        bbox = None
    else:
        x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
        face_roi = gray_eq[y:y+h, x:x+w]
        bbox = (x, y, w, h)

    if return_bbox:
        return face_roi, bbox
    else:
        return face_roi


def predict_emotion_from_bgr(model, bgr_frame: np.ndarray):
    """
    Mengembalikan:
    - label (str) atau None
    - confidence (float)
    - probs (np.ndarray)
    - bbox (x,y,w,h) atau None
    """
    face_roi, bbox = preprocess_and_detect_face(bgr_frame, return_bbox=True)

    if face_roi is None or face_roi.size == 0:
        return None, 0.0, np.zeros(len(CLASS_LABELS)), None

    pil_img = Image.fromarray(face_roi)
    img = transform(pil_img)
    img_tensor = cast(Tensor, img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        probs_t = F.softmax(logits, dim=1)
        probs = probs_t.cpu().numpy()[0]
        top_idx = int(np.argmax(probs))
        label = CLASS_LABELS[top_idx]
        confidence = float(probs[top_idx])

    return label, confidence, probs, bbox


# ----------------------------
# RINGKASAN EMOSI DOMINAN
# ----------------------------

def generate_emotion_summary(history_df: pd.DataFrame) -> str:
    if history_df.empty:
        return "Belum ada data emosi yang tercatat. Jalankan deteksi terlebih dahulu."

    counts = history_df["label"].value_counts(normalize=True)

    sad_ratio = counts.get("Sadness", 0.0)
    anger_ratio = counts.get("Anger", 0.0)
    happy_ratio = counts.get("Happy", 0.0)
    surprise_ratio = counts.get("Surprise", 0.0)

    neg_ratio = sad_ratio + anger_ratio
    pos_ratio = happy_ratio + surprise_ratio

    lines = []
    lines.append(f"- Dominasi emosi sedih (Sadness): {sad_ratio*100:.1f}%")
    lines.append(f"- Dominasi emosi marah (Anger): {anger_ratio*100:.1f}%")
    lines.append(f"- Dominasi emosi positif (Happy + Surprise): {pos_ratio*100:.1f}%")

    if neg_ratio >= 0.5:
        lines.append(
            "\n⚠️ **Peringatan**: Emosi negatif (sedih/marah) cukup dominan. "
            "Disarankan untuk melakukan pendekatan suportif, misalnya: "
            "istirahat sejenak, diskusi dengan atasan/HR, atau konseling jika diperlukan."
        )
    elif pos_ratio >= 0.5:
        lines.append(
            "\n✅ **Positif**: Emosi positif (senang/surprise menyenangkan) relatif dominan. "
            "Kondisi ini biasanya mendukung fokus, kreativitas, dan kualitas kerja yang baik."
        )
    else:
        lines.append(
            "\nℹ️ **Netral/Campuran**: Emosi yang terdeteksi cukup bervariasi. "
            "Perlu pemantauan berkala untuk melihat tren dalam jangka waktu lebih panjang."
        )

    return "\n".join(lines)


# ----------------------------
# VIDEO PROCESSOR (KAMERA)
# ----------------------------

class EmotionVideoProcessor(VideoTransformerBase):

    def __init__(self, model, employee_name: str, shift_label: str, log_every_n_frames: int = 15):
        self.model = model
        self.employee_name = employee_name or "Tanpa Nama"
        self.shift_label = shift_label
        self.log_every_n_frames = log_every_n_frames

        self.frame_counter = 0
        self.last_label = None
        self.last_confidence = 0.0
        self.last_probs = np.zeros(len(CLASS_LABELS))

    def recv(self, frame):
        img_bgr = frame.to_ndarray(format="bgr24")

        # Prediksi emosi frame ini
        label, confidence, probs, bbox = predict_emotion_from_bgr(self.model, img_bgr)

        self.last_label = label
        self.last_confidence = confidence
        self.last_probs = probs

        self.frame_counter += 1

        if label is not None:
            # Overlay teks + box wajah
            overlay_text = f"{label} ({confidence*100:.1f}%)"
            cv2.rectangle(img_bgr, (0, 0), (img_bgr.shape[1], 40), (0, 0, 0), -1)
            cv2.putText(
                img_bgr,
                overlay_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            if bbox is not None:
                x, y, w, h = bbox  # pyright: ignore[reportGeneralTypeIssues]
                cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # === LOG OTOMATIS TIAP N FRAME (kamera) ===
            if self.frame_counter % self.log_every_n_frames == 0:
                log_prediction(
                    "Realtime",
                    self.employee_name,
                    self.shift_label,
                    label,
                    confidence,
                )

        return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")


# ----------------------------
# PROSES VIDEO UPLOAD + LOG KE ANALITIK
# ----------------------------

def process_uploaded_video_bytes(
    video_bytes: bytes,
    model,
    source: str,
    employee_name: str,
    shift: str,
    progress_placeholder=None,
) -> str:
    """
    Proses file video yang diupload:
    - Membaca frame
    - Memprediksi emosi per frame
    - Menambahkan overlay teks emosi
    - Menyimpan ke file video baru
    - Mencatat hasil ke history (untuk analitik)
    """
    tfile_in = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile_in.write(video_bytes)
    tfile_in.close()

    cap = cv2.VideoCapture(tfile_in.name)
    if not cap.isOpened():
        st.error("Tidak dapat membuka video yang diupload.")
        return ""

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cv2.CAP_PROP_FRAME_COUNT or 1)

    fourcc = int(cv2.VideoWriter_fourcc(*"mp4v"))  # type: ignore[attr-defined]
    tfile_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out = cv2.VideoWriter(tfile_out.name, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        label, confidence, _, bbox = predict_emotion_from_bgr(model, frame)

        if label is not None:
            overlay_text = f"{label} ({confidence*100:.1f}%)"
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
            cv2.putText(
                frame,
                overlay_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            if bbox is not None:
                x, y, w, h = bbox  # pyright: ignore[reportGeneralTypeIssues]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # CATAT KE RIWAYAT UNTUK ANALITIK (source: Video)
            log_prediction(source, employee_name, shift, label, confidence)

        out.write(frame)

        if progress_placeholder is not None and frame_count:
            progress_placeholder.progress(min(frame_idx / frame_count, 1.0))

    cap.release()
    out.release()

    return tfile_out.name


# ----------------------------
# HALAMAN: INFERENCE & MONITORING
# ----------------------------

def render_inference_page(model):
    ensure_history_state()
    history_df = pd.DataFrame(st.session_state.history)

    total_preds = int(len(history_df))
    positive_rate = (
        f"{(history_df['label'].isin(POSITIVE_EMOTIONS).mean() * 100):.1f}%" if total_preds else "-"
    )
    avg_conf = f"{(history_df['confidence'].mean() * 100):.1f}%" if total_preds else "-"

    kpi_col_1, kpi_col_2, kpi_col_3 = st.columns(3)
    kpi_col_1.metric("Total Pengukuran", total_preds)
    kpi_col_2.metric("Rasio Emosi Positif", positive_rate)
    kpi_col_3.metric("Rata-rata Confidence", avg_conf)

    st.markdown("### Pengaturan Karyawan")
    employee_name = st.text_input("Nama Karyawan", value="", placeholder="Contoh: Andi Saputra")
    shift_label = st.selectbox("Shift", ["Pagi", "Siang", "Malam"], index=0)
    confidence_threshold = st.slider(
        "Peringatan bila probabilitas di bawah",
        0.0, 1.0, 0.6, 0.05
    )

    st.markdown("---")
    tab_cam, tab_video = st.tabs(["Kamera Real-time", "Upload Video"])

    # ----------- TAB KAMERA -----------
    with tab_cam:
        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.subheader("1. Monitor Kamera Real-time")
            st.write("Klik Start untuk mulai deteksi emosi langsung dari kamera Anda.")
            webrtc_ctx = webrtc_streamer(
                key="emotion_stream",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={"video": True, "audio": False},
                video_processor_factory=lambda: EmotionVideoProcessor(
                    model=model,
                    employee_name=employee_name.strip(),
                    shift_label=shift_label,
                    log_every_n_frames=30,
                ),
            )

        with col_right:
            st.subheader("Hasil Deteksi Terakhir (Kamera)")
            processor = webrtc_ctx.video_processor if webrtc_ctx and webrtc_ctx.state.playing else None
            if processor and processor.last_label:
                st.metric("Emosi", processor.last_label, f"{processor.last_confidence*100:.2f}%")
                if processor.last_confidence < confidence_threshold:
                    st.warning(
                        "Kepercayaan model rendah. Atur pencahayaan atau posisi wajah untuk hasil lebih stabil."
                    )
            elif webrtc_ctx and webrtc_ctx.state.playing:
                st.info("Model sedang memuat prediksi...")
            else:
                st.info("Mulai kamera untuk melihat hasil prediksi.")

            st.subheader("Distribusi Probabilitas (Frame Terakhir)")
            if processor and processor.last_label:
                prob_chart = px.bar(
                    x=CLASS_LABELS,
                    y=processor.last_probs,
                    labels={"x": "Emosi", "y": "Probabilitas"},
                    title="Probabilitas Emosi Saat Ini",
                )
                st.plotly_chart(prob_chart, use_container_width=True)
            else:
                st.info("Probabilitas akan muncul setelah kamera aktif dan mendeteksi wajah.")

    # ----------- TAB VIDEO UPLOAD -----------
    with tab_video:
        st.subheader("2. Deteksi Emosi dari Video")
        uploaded_video = st.file_uploader(
            "Upload video (mis. mp4) yang berisi ekspresi emosi karyawan",
            type=["mp4", "avi", "mov", "mkv"],
        )

        if uploaded_video is not None:
            video_bytes = uploaded_video.read()
            st.video(video_bytes)

            if st.button("Proses Video", use_container_width=True):
                progress = st.progress(0.0)
                with st.spinner("Memproses video..."):
                    out_path = process_uploaded_video_bytes(
                        video_bytes,
                        model,
                        source="Video",
                        employee_name=employee_name.strip(),
                        shift=shift_label,
                        progress_placeholder=progress,
                    )
                if out_path:
                    st.success("Video dengan anotasi emosi berhasil dibuat:")
                    with open(out_path, "rb") as f:
                        st.video(f.read())
                else:
                    st.error("Gagal memproses video.")

    # ----------- ANALITIK & KESIMPULAN -----------
    st.markdown("---")
    st.subheader("3. Analitik Emosi & Kesimpulan")

    history_df = pd.DataFrame(st.session_state.history)
    if not history_df.empty:
        emotion_counts = history_df['label'].value_counts().reset_index()
        emotion_counts.columns = ['Emosi', 'Jumlah']
        dist_chart = px.bar(emotion_counts, x='Emosi', y='Jumlah', title='Distribusi Emosi Terkumpul')
        st.plotly_chart(dist_chart, use_container_width=True)

        col_a, col_b, col_c = st.columns(3)
        positive_total = int(history_df['label'].isin(POSITIVE_EMOTIONS).sum())
        col_a.metric("Prediksi Positif", positive_total)
        col_b.metric("Prediksi Negatif", int(len(history_df) - positive_total))
        col_c.metric("Confidence Maks", f"{history_df['confidence'].max()*100:.2f}%")

        st.subheader("Riwayat Prediksi")
        st.dataframe(history_df.sort_values('timestamp', ascending=False), use_container_width=True)

        csv_data = history_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Unduh CSV Riwayat",
            data=csv_data,
            file_name="emotion_history.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.subheader("Kesimpulan Emosi Selama Pemantauan")
        summary_text = generate_emotion_summary(history_df)
        if "Peringatan" in summary_text:
            st.warning(summary_text)
        elif "Positif" in summary_text:
            st.success(summary_text)
        else:
            st.info(summary_text)
    else:
        st.info("Belum ada data analitik. Jalankan prediksi dan simpan hasil untuk mulai mengisi dashboard.")


# ----------------------------
# HALAMAN: INSIGHT EMOSI vs KINERJA
# ----------------------------

def render_article_page():
    st.header("Pengaruh Emosi Karyawan terhadap Kualitas Pekerjaan")
    st.write(ARTICLE_TEXT)


# ----------------------------
# HALAMAN: ABOUT
# ----------------------------

def render_about_page():
    st.header("About This Project & Team")
    st.markdown("""
**Judul Proyek**  
Deteksi Emosi Karyawan Berbasis Wajah Menggunakan DenseNet-169

**Tim Pengembang**  
- Muhammad Makarim (225150207111122) – Model & Dashboard  
- Rakha Alif Athallah (225150207111050) – Dataset, Model, & Poster.
    """)


# ----------------------------
# MAIN
# ----------------------------

def main():
    st.set_page_config(page_title="Emotion Recognition Dashboard", layout="wide")
    st.title("Emotion Recognition Dashboard")
    st.caption("Monitoring emosi karyawan menggunakan model DenseNet")

    try:
        model = load_model()
    except Exception as exc:
        st.error(f"Gagal memuat model: {exc}")
        st.stop()

    page = st.sidebar.radio(
        "Navigasi",
        ["Inference & Monitoring", "Insight Emosi vs Kinerja", "About"]
    )

    if page == "Inference & Monitoring":
        render_inference_page(model)
    elif page == "Insight Emosi vs Kinerja":
        render_article_page()
    else:
        render_about_page()


if __name__ == "__main__":
    main()
