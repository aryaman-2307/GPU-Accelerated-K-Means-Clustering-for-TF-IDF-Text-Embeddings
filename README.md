# 🚀 GPU-Accelerated K-Means Clustering for Text Embeddings

This project implements **K-Means clustering for text documents** using **TF-IDF embeddings** and **GPU acceleration via CuPy**. Designed to handle large-scale document clustering efficiently, it showcases the use of CUDA-enabled GPUs to speed up unsupervised learning on textual data.

---

## 📌 Key Features

- ✅ Vectorizes input documents using TF-IDF
- ⚡ Runs K-Means clustering on the GPU using CuPy (CUDA)
- 🔁 Supports a configurable number of clusters and iterations
- 🧮 Outputs cluster labels for each document via the command line
- 🛠 Easily extendable to use GloVe, BERT, or MiniBatch K-Means

---

## 🖥 Sample Output

```
Doc 1: Cluster 0  
Doc 2: Cluster 1  
Doc 3: Cluster 0  
...
```

---

## 📂 Project Structure

```
gpu_kmeans_text/
├── main.py             # Main runner: loads input, runs clustering
├── preprocess.py       # TF-IDF vectorization + normalization
├── kmeans_gpu.py       # CUDA-accelerated K-Means using CuPy
├── text_corpus.txt     # Sample input text file
├── requirements.txt    # Dependency list
└── README.md           # This file
```

---

## 🧪 Input Format

The input should be a plain `.txt` file with one document per line.

**Example:**
```
Apple launches new AI-powered iPhone.
Delicious recipes for homemade Italian pasta.
Wall Street sees gains in the tech sector this quarter.
```

---

## 🚀 Getting Started

### 1️⃣ Install Dependencies

> Ensure you have a CUDA-enabled GPU and a compatible CUDA toolkit.

```bash
pip install -r requirements.txt
```

If using CUDA 12.x (recommended for CUDA 12.4 users):
```bash
pip install cupy-cuda12x
```

> If you get DLL load errors, refer to the CuPy [Installation Guide](https://docs.cupy.dev/en/stable/install.html).

---

### 2️⃣ Run the Program

```bash
python main.py --input_file text_corpus.txt --k 3 --iter 100
```

**Arguments:**
- `--input_file`: Path to your document corpus (`.txt`)
- `--k`: Number of clusters (default: 3)
- `--iter`: Max iterations for convergence (default: 100)

---

## 🛠 Dependencies

```
cupy-cuda12x
numpy
scikit-learn
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 📊 Performance Tip

For larger datasets or more accurate clustering:
- Replace TF-IDF with sentence embeddings (e.g., BERT)
- Visualize clusters using PCA or t-SNE
- Compare performance with a CPU-based K-Means baseline

---

## 📸 Proof of Execution

- Console output of document clusters
- Optional: output `.csv` for logs
- Use `nvidia-smi` to confirm GPU usage

---

## 💡 Future Improvements

- [ ] Add CSV export for results
- [ ] Add t-SNE visualizations
- [ ] Integrate deep language models (e.g., BERT or RoBERTa)
- [ ] Create an interactive dashboard for cluster exploration

---
