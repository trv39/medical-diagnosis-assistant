# Medical Diagnosis Assistant
### YOLO-based Chest X-ray Anomaly Detection with PubMed Literature Retrieval

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://trv-medical-diagnosis-assistant-zypaymxqeegxmxbuavtayy.streamlit.app/)

## Overview
An end-to-end medical AI system that detects chest X-ray anomalies using 
YOLOv11n and retrieves relevant research papers from a 43k+ PubMed corpus 
using FAISS semantic search.

## Architecture
```
Chest X-ray Image
      ↓
YOLOv11n (ONNX) — 14-class anomaly detection
      ↓
Query Formation
      ↓
FAISS Similarity Search — 43k PubMed papers
      ↓
Top-K Relevant Papers returned
```

## Tech Stack
| Component | Technology |
|---|---|
| Anomaly Detection | YOLOv11n (ONNX, CPU) |
| Dataset | VinBigData Chest X-ray (15k images, 14 classes) |
| Literature Corpus | PubMed (43,273 papers scraped) |
| Embeddings | all-MiniLM-L6-v2 |
| Vector Search | FAISS (IndexFlatIP) |
| Evaluation | BERTScore (distilbert-base-uncased) |
| Frontend | Streamlit |
| Scraping | BeautifulSoup + NCBI E-utilities API |

## Results
| Metric | Value |
|---|---|
| mAP50 | 0.33 |
| mAP50-95 | 0.163 |
| Cardiomegaly AP | 0.686 |
| Aortic Enlargement AP | 0.668 |
| PubMed Papers Indexed | 43,273 |
| FAISS Retrieval Score | 0.63-0.69 |

## Classes Detected
Aortic Enlargement, Atelectasis, Calcification, Cardiomegaly,
Consolidation, ILD, Infiltration, Lung Opacity, Nodule/Mass,
Other Lesion, Pleural Effusion, Pleural Thickening, Pneumothorax,
Pulmonary Fibrosis

## Live Demo
[Click here to try the app](https://trv-medical-diagnosis-assistant-zypaymxqeegxmxbuavtayy.streamlit.app/)

## Project Structure
```
├── app.py                 # Streamlit frontend
├── best.onnx              # YOLOv11n ONNX model
├── pubmed.index           # FAISS vector index
├── metadata.pkl.gz        # PubMed paper metadata
├── requirements.txt       # Dependencies
└── bertscore_results.csv  # Evaluation results
```

## Setup
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Methodology
1. **Data Collection**: Scraped 43k+ PubMed abstracts across 15 chest 
   disease queries using BeautifulSoup and NCBI E-utilities API
2. **YOLO Training**: Fine-tuned YOLOv11n on VinBigData dataset with 
   real radiologist bounding box annotations
3. **FAISS Indexing**: Embedded all papers using sentence-transformers 
   and indexed with FAISS for fast similarity search
4. **Evaluation**: BERTScore used to evaluate semantic relevance of 
   retrieved papers against clinical reference descriptions
