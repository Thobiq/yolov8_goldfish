import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="Deteksi Penyakit Ikan Mas Koki", page_icon="🐠")

st.title("Deteksi Penyakit Ikan Mas Koki")
st.write("Upload foto ikan mas koki Anda, dan AI akan mendeteksi apakah ada penyakit pada ikan Anda.")

@st.cache_resource
def load_model():
    model = YOLO('best.pt') 
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}. Pastikan file 'best.pt' ada di folder yang sama.")

uploaded_file = st.file_uploader("Pilih gambar ikan...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Gambar Asli")
        st.image(image, use_container_width=True)

    if st.button("Deteksi Penyakit"):
        with st.spinner('Sedang menganalisa...'):
            results = model.predict(image, conf=0.25)
            
            res_plotted = results[0].plot() 
            
            with col2:
                st.subheader("Hasil Deteksi")
                st.image(res_plotted, channels="BGR", use_container_width=True) 
            
            st.success("Analisa Selesai!")
            
            boxes = results[0].boxes
            if len(boxes) > 0:
                with st.expander("Lihat Rincian Deteksi"):
                    for box in boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        name = model.names[cls]
                        st.write(f"- Terdeteksi: **{name}** (Kepercayaan: {conf:.2f})")
            else:
                st.warning("Tidak ada penyakit atau ikan yang terdeteksi.")