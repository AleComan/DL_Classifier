import requests
import streamlit as st
from PIL import Image
import io

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Real Estate Classifier",
    page_icon="🏠",
    layout="centered",
)

st.title("🏠 Real Estate Image Classifier")
st.markdown("Sube una imagen inmobiliaria y el modelo la clasificará automáticamente.")

# ── Sidebar con info ─────────────────────────────────────────
with st.sidebar:
    st.header("Estado de la API")
    try:
        response = requests.get(f"{API_URL}/", timeout=3)
        info = response.json()
        st.success("API conectada")
        st.json(info)
    except Exception:
        st.error("API no disponible — arranca FastAPI primero")

    st.header("Clases disponibles")
    try:
        classes = requests.get(f"{API_URL}/classes", timeout=3).json()["classes"]
        for c in classes:
            st.markdown(f"- {c}")
    except Exception:
        st.warning("No se pudieron cargar las clases")

# ── Upload ───────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Selecciona una imagen",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Imagen subida", use_container_width=True)

    if st.button("Clasificar", type="primary"):
        with st.spinner("Clasificando..."):
            try:
                # Reenviar imagen a la API
                img_bytes = io.BytesIO()
                image.save(img_bytes, format="JPEG")
                img_bytes.seek(0)

                response = requests.post(
                    f"{API_URL}/predict",
                    files={"file": ("image.jpg", img_bytes, "image/jpeg")},
                    timeout=10,
                )
                response.raise_for_status()
                result = response.json()

            except requests.exceptions.ConnectionError:
                st.error("No se puede conectar con la API. ¿Está arrancada?")
                st.stop()
            except Exception as e:
                st.error(f"Error en la clasificación: {e}")
                st.stop()

        # ── Resultado principal ──────────────────────────────
        st.success(f"**{result['prediction']}**")
        confidence = result["confidence"] * 100
        st.metric("Confianza", f"{confidence:.1f}%")

        # ── Top 5 ────────────────────────────────────────────
        st.subheader("Top 5 predicciones")
        for item in result["top5"]:
            prob = item["probability"] * 100
            st.markdown(f"**{item['class']}**")
            st.progress(item["probability"], text=f"{prob:.1f}%")