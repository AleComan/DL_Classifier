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

st.title("🏠 DL Classifier")
st.markdown("Sube una o varias imágenes y el modelo las clasificará automáticamente.")

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

# ── Upload múltiple ──────────────────────────────────────────
uploaded_files = st.file_uploader(
    "Selecciona una o varias imágenes",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.markdown(f"**{len(uploaded_files)} imagen(es) cargada(s)**")

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
        st.subheader(f"Resultados — {len(results)} imagen(es) procesada(s)")

        cols_per_row = 3
        for row_start in range(0, len(results), cols_per_row):
            cols = st.columns(cols_per_row)
            for col, item in zip(cols, results[row_start:row_start + cols_per_row]):
                with col:
                    st.image(item["image"], use_column_width=True)
                    r = item["result"]
                    confidence = r["confidence"] * 100
                    st.markdown(f"**{r['prediction']}** — `{confidence:.1f}%`")
                    st.caption(item["name"])
                    with st.expander("Top 5"):
                        for entry in r["top5"]:
                            prob = entry["probability"] * 100
                            st.progress(entry["probability"], text=f"{entry['class']} — {prob:.1f}%")

        # ── Errores si los hay ───────────────────────────────
        if errors:
            st.divider()
            st.warning(f"{len(errors)} imagen(es) no se pudieron procesar:")
            for e in errors:
                st.markdown(f"- `{e['name']}`: {e['error']}")