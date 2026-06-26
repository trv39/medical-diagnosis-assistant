
import streamlit as st
import onnxruntime as ort
import numpy as np
import faiss
import pickle
import gzip
from PIL import Image, ImageDraw
from sentence_transformers import SentenceTransformer
import os
import tempfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""

CLASS_NAMES = [
    "Aortic_enlargement", "Atelectasis", "Calcification",
    "Cardiomegaly", "Consolidation", "ILD", "Infiltration",
    "Lung_Opacity", "Nodule_Mass", "Other_lesion",
    "Pleural_effusion", "Pleural_thickening", "Pneumothorax",
    "Pulmonary_fibrosis"
]

DISEASE_INFO = {
    "Aortic_enlargement": "Widening of the aorta beyond normal diameter, may indicate aneurysm or hypertension.",
    "Atelectasis": "Partial or complete collapse of lung tissue causing reduced gas exchange.",
    "Calcification": "Calcium deposits in lung tissue, often indicating healed infection or granuloma.",
    "Cardiomegaly": "Enlargement of the heart beyond normal size, associated with heart failure.",
    "Consolidation": "Lung airspace filled with fluid instead of air, seen in pneumonia.",
    "ILD": "Interstitial lung disease causing scarring and stiffening of lung tissue.",
    "Infiltration": "Abnormal substance filling the lung airspace, seen in infection or inflammation.",
    "Lung_Opacity": "Area of increased density in lung tissue indicating fluid or tissue abnormality.",
    "Nodule_Mass": "Abnormal round growth in the lung, may be benign or malignant.",
    "Other_lesion": "Miscellaneous lung abnormality not classified in other categories.",
    "Pleural_effusion": "Abnormal fluid accumulation in the pleural space surrounding the lung.",
    "Pleural_thickening": "Thickening of the pleural membrane often due to inflammation or fibrosis.",
    "Pneumothorax": "Air in pleural space causing partial or complete lung collapse.",
    "Pulmonary_fibrosis": "Scarring and thickening of lung tissue causing progressive breathing difficulty."
}

INPUT_SIZE = 640
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_models():
    sess_options = ort.SessionOptions()
    sess_options.inter_op_num_threads = 4
    sess_options.intra_op_num_threads = 4
    onnx_sess = ort.InferenceSession(
        os.path.join(BASE_DIR, "best.onnx"),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"]
    )
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index(os.path.join(BASE_DIR, "pubmed.index"))
    with gzip.open(os.path.join(BASE_DIR, "metadata.pkl.gz"), "rb") as f:
        metadata = pickle.load(f)
    return onnx_sess, embedder, index, metadata

def run_inference(onnx_sess, image_path, conf_threshold):
    img = Image.open(image_path).convert("RGB").resize((INPUT_SIZE, INPUT_SIZE))
    arr = np.array(img).astype("float32") / 255.0
    arr = arr.transpose(2, 0, 1)
    arr = np.expand_dims(arr, axis=0)
    input_name = onnx_sess.get_inputs()[0].name
    outputs = onnx_sess.run(None, {input_name: arr})
    predictions = outputs[0][0]

    detections = []
    for pred in predictions.T:
        x_center, y_center, width, height = pred[:4]
        class_scores = pred[4:]
        cls_idx = int(np.argmax(class_scores))
        conf = float(class_scores[cls_idx])
        if conf >= conf_threshold:
            detections.append({
                "class": CLASS_NAMES[cls_idx],
                "confidence": round(conf, 4),
                "bbox": [
                    round(float(x_center - width / 2), 4),
                    round(float(y_center - height / 2), 4),
                    round(float(x_center + width / 2), 4),
                    round(float(y_center + height / 2), 4)
                ]
            })

    def iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
        area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0

    detections.sort(key=lambda x: x["confidence"], reverse=True)
    final = []
    for det in detections:
        keep = True
        for kept in final:
            if det["class"] == kept["class"] and iou(det["bbox"], kept["bbox"]) > 0.45:
                keep = False
                break
        if keep:
            final.append(det)
    return final

def draw_boxes(image_path, detections):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    colors = ["#FF4444", "#FF8800", "#FFCC00", "#44FF44", "#4444FF",
              "#FF44FF", "#44FFFF", "#FF4488", "#88FF44", "#4488FF",
              "#FF8844", "#44FF88", "#8844FF", "#FF4444"]
    for i, det in enumerate(detections):
        color = colors[i % len(colors)]
        x1, y1, x2, y2 = det["bbox"]
        draw.rectangle([x1*w, y1*h, x2*w, y2*h], outline=color, width=3)
        label = f"{det['class'].replace('_', ' ')} {det['confidence']:.0%}"
        draw.rectangle([x1*w, y1*h - 18, x1*w + len(label)*7, y1*h], fill=color)
        draw.text((x1*w + 2, y1*h - 16), label, fill="white")
    return img

def retrieve_papers(query, embedder, index, metadata, top_k):
    q_emb = embedder.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(metadata):
            doc = metadata[idx].copy()
            doc["similarity_score"] = round(float(score), 4)
            results.append(doc)
    return results

st.set_page_config(
    page_title="Medical Diagnosis Assistant",
    page_icon="🫁",
    layout="wide"
)

st.title("🫁 Medical Diagnosis Assistant")
st.markdown("**YOLOv11n Chest X-ray Anomaly Detection + PubMed Literature Retrieval**")
st.markdown("---")

with st.spinner("Loading models..."):
    onnx_sess, embedder, index, metadata = load_models()

with st.sidebar:
    st.header("Settings")
    conf_threshold = st.slider("Detection Confidence", 0.10, 0.90, 0.25, 0.05)
    top_k = st.slider("Papers to Retrieve", 1, 10, 5)
    st.markdown("---")
    st.markdown("**Stack**")
    st.markdown("- YOLOv11n (ONNX)")
    st.markdown("- FAISS Vector Search")
    st.markdown("- PubMed 43k Papers")
    st.markdown("- Sentence Transformers")

uploaded_file = st.file_uploader("Upload Chest X-ray", type=["jpg", "jpeg", "png"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original X-ray")
        st.image(tmp_path, use_container_width=True)

    with st.spinner("Running YOLO detection..."):
        detections = run_inference(onnx_sess, tmp_path, conf_threshold)

    with col2:
        st.subheader("Detection Results")
        if detections:
            annotated = draw_boxes(tmp_path, detections)
            st.image(annotated, use_container_width=True)
        else:
            st.image(tmp_path, use_container_width=True)
            st.info("No abnormalities detected above threshold.")

    st.markdown("---")

    if detections:
        st.subheader("Detected Conditions")
        cols = st.columns(min(len(detections), 3))
        for i, det in enumerate(detections):
            with cols[i % 3]:
                conf_color = "red" if det["confidence"] > 0.7 else "orange" if det["confidence"] > 0.4 else "blue"
                st.markdown(f"**:{conf_color}[{det['class'].replace('_', ' ')}]**")
                st.progress(det["confidence"])
                st.caption(f"Confidence: {det['confidence']:.2%}")
                st.caption(DISEASE_INFO.get(det["class"], ""))
        query = " ".join([d["class"].replace("_", " ") for d in detections])
        query += " chest X-ray diagnosis findings treatment"
    else:
        query = "normal chest X-ray findings radiology"

    st.markdown("---")
    st.subheader("Retrieved Medical Literature")
    st.caption(f"Query: `{query}`")

    with st.spinner("Searching PubMed corpus..."):
        papers = retrieve_papers(query, embedder, index, metadata, top_k)

    for i, paper in enumerate(papers):
        with st.expander(f"[{i+1}] {paper['title']} ({paper['year']}) — Score: {paper['similarity_score']}"):
            col_a, col_b = st.columns([1, 3])
            with col_a:
                st.markdown(f"**PMID:** `{paper['pmid']}`")
                st.markdown(f"**Year:** {paper['year']}")
                st.markdown(f"[PubMed Link](https://pubmed.ncbi.nlm.nih.gov/{paper['pmid']}/)")
            with col_b:
                st.markdown("**Abstract:**")
                st.write(paper["abstract"])

    os.unlink(tmp_path)

else:
    st.info("Upload a chest X-ray image to get started.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**1. Upload**")
        st.markdown("Upload any chest X-ray in JPG or PNG format.")
    with col2:
        st.markdown("**2. Detect**")
        st.markdown("YOLOv11n detects anomalies across 14 chest conditions.")
    with col3:
        st.markdown("**3. Retrieve**")
        st.markdown("FAISS searches 43k+ PubMed papers for relevant literature.")
