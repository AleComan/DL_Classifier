import base64
import requests
import streamlit as st
from PIL import Image
import io

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="DL Classifier",
    page_icon="🏠",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    /* Forzar esquema de color claro */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
        color-scheme: light !important;
    }

    /* Importar fuente moderna */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Contenido principal — más ancho para la grid de 3 */
    .block-container {
        max-width: 960px !important;
        margin-left: auto !important;
        margin-right: auto !important;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Título */
    h1 {
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem !important;
        text-align: center;
    }

    /* Subtítulo */
    .subtitle {
        color: #6b7280;
        font-size: 0.95rem;
        margin-bottom: 1.75rem;
        text-align: center;
    }

    /* Centrar uploader */
    [data-testid="stFileUploader"] {
        display: flex;
        flex-direction: column;
        align-items: center;
    }

    /* Badge de conteo de imágenes */
    .img-count-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        color: #1d4ed8;
        font-size: 0.85rem;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 20px;
        margin: 0.5rem auto 1rem auto;
    }

    /* Centrar badge y botón */
    .main .stMarkdown { text-align: center; }
    .main div.stButton { display: flex; justify-content: center; }

    /* Centrar subheader de resultados */
    h2, h3 { text-align: center; }

    /* ── Tarjeta de resultado ── */
    .result-card {
        color-scheme: light;
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        margin-bottom: 1rem;
        transition: box-shadow 0.2s;
    }
    .result-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.12); }

    .card-img {
        width: 100%;
        height: 200px;
        object-fit: cover;
        display: block;
    }

    .card-body {
        padding: 0.75rem 0.9rem 0.65rem 0.9rem;
    }

    /* Fila predicción + badge confianza */
    .card-pred-row {
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
        margin-bottom: 3px;
    }

    .pred-label {
        font-size: 0.95rem;
        font-weight: 600;
        color: #111827;
    }

    /* Badges de confianza con color semántico */
    .conf-badge {
        font-size: 0.75rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 20px;
    }
    .conf-high { background: #d1fae5; color: #065f46; }
    .conf-mid  { background: #fef3c7; color: #92400e; }
    .conf-low  { background: #fee2e2; color: #991b1b; }

    /* Nombre del fichero */
    .filename {
        font-size: 0.72rem;
        color: #9ca3af;
        margin: 0;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        color-scheme: light;
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    [data-testid="stSidebar"] .stMarkdown p {
        font-size: 0.85rem;
        text-align: left;
    }
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        text-align: left;
        font-size: 0.78rem !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94a3b8 !important;
        margin-bottom: 0.6rem;
    }

    /* Card del modelo en sidebar */
    .model-card {
        color-scheme: light;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.7rem 0.85rem;
        margin-top: 0.5rem;
    }

    /* Accuracy badge grande */
    .acc-badge {
        display: inline-block;
        background: #d1fae5;
        color: #065f46;
        font-size: 1.1rem;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 8px;
        margin-bottom: 0.6rem;
    }

    /* Filas de info en sidebar */
    .info-table { color-scheme: light; width: 100%; border-collapse: collapse; font-size: 0.8rem; }
    .info-table td { padding: 3px 2px; vertical-align: top; }
    .info-table td:first-child {
        color: #94a3b8;
        white-space: nowrap;
        padding-right: 8px;
        font-weight: 500;
    }
    .info-table td:last-child {
        color: #1e293b;
        font-weight: 600;
        word-break: break-all;
    }

    /* Pills de clases */
    .class-pills {
        color-scheme: light;
        display: flex;
        flex-wrap: wrap;
        gap: 5px;
        margin-top: 0.4rem;
    }
    .class-pill {
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        color: #475569;
        font-size: 0.72rem;
        font-weight: 500;
        padding: 3px 9px;
        border-radius: 20px;
    }

    /* Botón primario */
    div.stButton > button[kind="primary"] {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 2rem;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    if st.button("⟳  Recargar modelo"):
        r = requests.post(f"{API_URL}/reload")
        if r.ok:
            st.success("Modelo recargado")
            st.rerun()
        else:
            st.error("Error al recargar")

    st.divider()

    st.header("Modelo en uso")
    try:
        response = requests.get(f"{API_URL}/", timeout=3)
        info = response.json()
        val_acc_pct = f"{info['val_acc'] * 100:.2f}%" if info.get('val_acc') else "—"
        st.markdown(
            f"""
            <div class="model-card">
              <div class="acc-badge">✓ {val_acc_pct}</div>
              <table class="info-table">
                <tr><td>Backbone</td><td>{info.get('model', '—')}</td></tr>
                <tr><td>Run ID</td><td>{info.get('run_id', '—')}</td></tr>
                <tr><td>Epoch</td><td>{info.get('epoch', '—')}</td></tr>
                <tr><td>Device</td><td>{info.get('device', '—')}</td></tr>
              </table>
            </div>
            """,
            unsafe_allow_html=True,
        )
    except Exception:
        st.error("API no disponible — arranca FastAPI primero")

    st.divider()

    st.header("Clases disponibles")
    try:
        classes = requests.get(f"{API_URL}/classes", timeout=3).json()["classes"]
        pills_html = '<div class="class-pills">' + "".join(
            f'<span class="class-pill">{c}</span>' for c in classes
        ) + '</div>'
        st.markdown(pills_html, unsafe_allow_html=True)
    except Exception:
        st.warning("No se pudieron cargar las clases")

# ── Header ───────────────────────────────────────────────────
st.title("🏠 DL Classifier")
st.markdown('<p class="subtitle">Sube una o varias imágenes y el modelo las clasificará automáticamente.</p>', unsafe_allow_html=True)

# ── Upload múltiple ──────────────────────────────────────────
uploaded_files = st.file_uploader(
    "Selecciona una o varias imágenes",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

if uploaded_files:
    n = len(uploaded_files)
    st.markdown(
        f'<div style="text-align:center"><span class="img-count-badge">🖼 {n} imagen{"es" if n > 1 else ""} lista{"s" if n > 1 else ""}</span></div>',
        unsafe_allow_html=True,
    )
    st.write("")

    if st.button("Clasificar todas", type="primary"):
        results = []
        errors = []

        progress = st.progress(0, text="Clasificando...")

        for i, uploaded in enumerate(uploaded_files):
            try:
                image = Image.open(uploaded).convert("RGB")
                img_bytes = io.BytesIO()
                image.save(img_bytes, format="JPEG")
                img_bytes.seek(0)

                response = requests.post(
                    f"{API_URL}/predict",
                    files={"file": (uploaded.name, img_bytes, "image/jpeg")},
                    timeout=10,
                )
                response.raise_for_status()
                results.append({
                    "name": uploaded.name,
                    "image": image,
                    "result": response.json(),
                })
            except requests.exceptions.ConnectionError:
                st.error("No se puede conectar con la API. ¿Está arrancada?")
                st.stop()
            except Exception as e:
                errors.append({"name": uploaded.name, "error": str(e)})

            progress.progress((i + 1) / len(uploaded_files), text=f"Clasificando {i + 1}/{len(uploaded_files)}...")

        progress.empty()

        # ── Resultados en grid ───────────────────────────────
        st.divider()
        st.subheader(f"Resultados — {len(results)} imagen(es) procesada(s)")
        st.write("")

        cols_per_row = min(len(results), 3)
        for row_start in range(0, len(results), cols_per_row):
            batch = results[row_start:row_start + cols_per_row]
            if len(batch) == 1:
                _, mid, _ = st.columns([1, 2, 1])
                cols = [mid]
            else:
                cols = st.columns(cols_per_row, gap="medium")

            for col, item in zip(cols, batch):
                with col:
                    r = item["result"]
                    confidence = r["confidence"] * 100

                    # Encode image to base64 para embeberla dentro de la card
                    buf = io.BytesIO()
                    item["image"].save(buf, format="JPEG", quality=85)
                    img_b64 = base64.b64encode(buf.getvalue()).decode()

                    # Clase CSS del badge según confianza
                    if confidence >= 90:
                        badge_cls = "conf-high"
                    elif confidence >= 70:
                        badge_cls = "conf-mid"
                    else:
                        badge_cls = "conf-low"

                    st.markdown(
                        f"""
                        <div class="result-card">
                          <img class="card-img" src="data:image/jpeg;base64,{img_b64}" alt="{item['name']}"/>
                          <div class="card-body">
                            <div class="card-pred-row">
                              <span class="pred-label">{r["prediction"]}</span>
                              <span class="conf-badge {badge_cls}">{confidence:.1f}%</span>
                            </div>
                            <p class="filename">{item["name"]}</p>
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    with st.expander("Top 5"):
                        for entry in r["top5"]:
                            prob = entry["probability"] * 100
                            st.progress(entry["probability"], text=f"{entry['class']} — {prob:.1f}%")
                    st.write("")

        # ── Errores si los hay ───────────────────────────────
        if errors:
            st.divider()
            st.warning(f"{len(errors)} imagen(es) no se pudieron procesar:")
            for e in errors:
                st.markdown(f"- `{e['name']}`: {e['error']}")