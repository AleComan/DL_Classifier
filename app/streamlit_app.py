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
    /* Centrar y limitar ancho del contenido principal */
    .block-container {
        max-width: 700px !important;
        margin-left: auto !important;
        margin-right: auto !important;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    /* Título */
    h1 {
        font-size: 2rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.25rem !important;
        text-align: center;
    }

    /* Centrar el file uploader */
    [data-testid="stFileUploader"] {
        display: flex;
        flex-direction: column;
        align-items: center;
    }

    /* Centrar el texto de conteo de imágenes y el botón */
    .main .stMarkdown p {
        text-align: center;
    }
    .main div.stButton {
        display: flex;
        justify-content: center;
    }

    /* Centrar el subheader de resultados */
    h2, h3 {
        text-align: center;
    }

    /* Subtítulo bajo el título */
    .subtitle {
        color: #6b7280;
        font-size: 0.95rem;
        margin-bottom: 1.75rem;
    }

    /* Tarjeta de resultado */
    .result-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1rem 1rem 0.75rem 1rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        margin-bottom: 1rem;
    }

    /* Etiqueta de predicción */
    .pred-label {
        font-size: 1rem;
        font-weight: 600;
        color: #111827;
    }

    /* Porcentaje de confianza */
    .pred-conf {
        font-size: 0.9rem;
        color: #059669;
        font-weight: 600;
    }

    /* Nombre del fichero */
    .filename {
        font-size: 0.75rem;
        color: #9ca3af;
        margin-top: 0.1rem;
    }

    /* Limitar ancho de imagen cuando hay pocas columnas */
    [data-testid="stImage"] img {
        border-radius: 8px;
        max-height: 340px;
        object-fit: cover;
        width: 100%;
    }

    /* Sidebar más limpio */
    [data-testid="stSidebar"] {
        background-color: #f9fafb;
    }
    [data-testid="stSidebar"] .stMarkdown p {
        font-size: 0.85rem;
        text-align: left;
    }

    /* Botón primario */
    div.stButton > button[kind="primary"] {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.45rem 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    if st.button("Recargar modelo"):
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
        st.success("API conectada")
        st.markdown(f"**Backbone:** `{info['model']}`")
        st.markdown(f"**Val accuracy:** `{round(info['val_acc'], 3)}`")
        st.markdown(f"**Run ID:** `{info['run_id']}`")
        st.markdown(f"**Epoch:** `{info['epoch']}`")
        st.markdown(f"**Device:** `{info['device']}`")
    except Exception:
        st.error("API no disponible — arranca FastAPI primero")

    st.divider()

    st.header("Clases disponibles")
    try:
        classes = requests.get(f"{API_URL}/classes", timeout=3).json()["classes"]
        for c in classes:
            st.markdown(f"- {c}")
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
    st.markdown(f"**{len(uploaded_files)} imagen(es) lista(s)**")
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
            # Si hay una sola imagen en la fila, centrarla con columnas vacías laterales
            if len(batch) == 1:
                _, mid, _ = st.columns([1, 2, 1])
                cols = [mid]
            else:
                cols = st.columns(cols_per_row, gap="medium")
            for col, item in zip(cols, batch):
                with col:
                    r = item["result"]
                    confidence = r["confidence"] * 100

                    st.image(item["image"], use_container_width=True)
                    st.markdown(
                        f'<p class="pred-label">{r["prediction"]} '
                        f'<span class="pred-conf">{confidence:.1f}%</span></p>'
                        f'<p class="filename">{item["name"]}</p>',
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