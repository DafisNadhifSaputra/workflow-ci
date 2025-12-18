# Workflow CI - MLflow Training Pipeline
## Proyek Akhir: Membangun Sistem Machine Learning
### Dicoding Indonesia

---

**Nama**: Dafis Nadhif Saputra

---

## 📋 Deskripsi Proyek

Repository ini berisi **MLflow Project** dan **GitHub Actions CI/CD pipeline** untuk melatih model machine learning secara otomatis. Setiap kali ada perubahan pada kode, workflow akan:

1. Menjalankan training model
2. Menyimpan artifacts ke MLflow
3. Build Docker image (opsional)
4. Push ke Docker Hub (opsional)

## 📊 Dataset

- **Nama**: Breast Cancer Wisconsin (Diagnostic)
- **Task**: Binary Classification
- **Target**: Malignant (0) vs Benign (1)

## 📁 Struktur Repository

```
workflow-ci/
├── README.md
├── requirements.txt
├── MLProject                    # MLflow Project configuration
├── conda.yaml                   # Conda environment
├── modelling.py                 # Training dengan MLflow autolog
├── modelling_tuning.py          # Training dengan manual logging
├── Dafis-Nadhif-Saputra.py     # Preprocessing script
├── data/
│   └── breast_cancer_data.csv
└── .github/
    └── workflows/
        └── ci.yml               # GitHub Actions CI/CD
```

## 🚀 Cara Menjalankan

### 1. Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/workflow-ci.git
cd workflow-ci
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Jalankan Training (Langsung)
```bash
# Dengan MLflow autolog
python modelling.py

# Dengan manual logging
python modelling_tuning.py
```

### 4. Jalankan dengan MLflow Project
```bash
# Jalankan entry point default (training)
mlflow run . --env-manager=local

# Jalankan preprocessing
mlflow run . -e preprocess --env-manager=local

# Jalankan hyperparameter tuning
mlflow run . -e tune --env-manager=local
```

### 5. Lihat MLflow UI
```bash
mlflow ui --port 5000
# Buka http://localhost:5000
```

## 🐳 Docker

### Build Docker Image dari Model
```bash
# Setelah training, build Docker image
mlflow models build-docker -m "runs:/<RUN_ID>/model" -n smsml-model

# Jalankan container
docker run -p 5001:8080 smsml-model
```

### Test Prediction
```bash
curl -X POST http://localhost:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{"inputs": [[17.99, 10.38, 122.8, 1001, 0.1184, ...]]}'
```

## ⚙️ GitHub Actions CI/CD

Workflow otomatis akan berjalan ketika:
- Push ke branch `main` atau `master`
- Pull request ke branch utama
- Manual trigger via `workflow_dispatch`

### Workflow Steps:
1. Checkout code
2. Setup Python 3.12
3. Install dependencies
4. Run MLflow training
5. Upload artifacts
6. (Optional) Build & push Docker image

### Secrets yang Diperlukan (untuk Docker Hub):
- `DOCKER_USERNAME`: Username Docker Hub
- `DOCKER_TOKEN`: Access token Docker Hub

## 📈 MLflow Tracking

### Metrics yang Di-track:
- Accuracy, Precision, Recall, F1-Score
- Training time
- Model parameters

### Artifacts yang Disimpan:
- Trained model (sklearn)
- Scaler (StandardScaler)
- Confusion matrix plot
- Feature importance plot
- Classification report

## 📝 Kriteria Dicoding

✅ Membuat folder MLProject  
✅ Membuat Workflow CI dengan GitHub Actions  
✅ Menyimpan artefak ke repository  
✅ (Advanced) Build Docker Images ke Docker Hub  

---

## 🔗 Links

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Documentation](https://docs.docker.com/)

---

**© 2024 Dafis Nadhif Saputra - Dicoding Indonesia**
