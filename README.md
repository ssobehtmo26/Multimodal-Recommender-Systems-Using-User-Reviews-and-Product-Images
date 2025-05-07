# 🧠 Multimodal Recommender Systems Using User Reviews and Product Images
**Author: Om Choudhary**  
*Course: DA623 – Computing with Signals*

---

## 📘 Notebook

🔗 **[View Full Notebook](https://github.com/ssobehtmo26/Multimodal-Recommender-Systems-Using-User-Reviews-and-Product-Images/blob/main/multimodal_recommender.ipynb)**

---

## 🚀 Motivation
In an era where online shopping dominates, building recommender systems that go beyond simple numeric ratings or keyword matching has become essential.  
I chose this topic to explore how integrating **user-generated reviews** with **product images** can enhance rating prediction — foundational to personalized recommendations in fashion e-commerce.

---

## 🧰 Historical Context: Multimodal Learning in Recommendation

Recommender systems have evolved from:
- 🧩 **Collaborative filtering**
- 📝 **Content-based filtering**
- 🧠 **Multimodal learning** (text + images + audio, etc.)

> Recent research (e.g., Chen et al., 2020; McAuley et al., 2015) confirms that combining reviews and product images improves accuracy and cold-start performance.

---

## 🎓 Summary

### Step 1: Data Loading & Cleaning
- Loaded `Amazon_Fashion.jsonl` (reviews) and `meta_Amazon_Fashion.jsonl` (metadata)
- Sampled 25,000 entries, merged using `parent_asin`
- Extracted the best available product image URL

### Step 2: Image Download
- Downloaded 2,000 images via `requests` and `PIL`

### Step 3: Feature Extraction
- Text → `DistilBERT` embeddings (768 dims)
- Image → `ResNet50` embeddings (2048 dims)
- Final input → 2816-dim multimodal feature vector

### Step 4: Deep Learning Classifier
- 3 hidden layers (1024 → 512 → 128)
- BatchNorm + Dropout
- Optimized using `Adam` with learning rate scheduler
- Balanced labels using `class_weight`

### Step 5: Evaluation Metrics
- Accuracy, Confusion Matrix, Classification Report
- Visualization of training loss and accuracy

---

## 📊 Results

| Metric             | Value         |
|--------------------|---------------|
| ✅ Accuracy         | **60%**     |
| 📊 Best Class       | 5-star ratings |
| 🧪 Observations     | Better performance on 4-5 stars; difficult to differentiate mid-tier (3-star) |

![Screenshot 2025-05-07 140211](https://github.com/user-attachments/assets/26eb18cc-7371-4e9c-9a9f-b9aeed234a9a)

![Screenshot 2025-05-07 140229](https://github.com/user-attachments/assets/e3c27e54-cc85-4509-9990-8832c56f2a6f)

![Screenshot 2025-05-07 140243](https://github.com/user-attachments/assets/b1b1da02-4b1d-4b63-b5a7-54022c26fee1)

![Screenshot 2025-05-07 140249](https://github.com/user-attachments/assets/1e0d9d8a-5c79-42e5-bab7-cd73feaf1db5)

![Screenshot 2025-05-07 140259](https://github.com/user-attachments/assets/0a9f0ed7-5cd9-4419-afe9-cd389970f152)
---

## 🌍 Key Learnings
- Multimodal input outperforms single-modality models
- DistilBERT captures sentiment better than TF-IDF
- Image features boost accuracy, especially for visual items

---

## 🧐 Reflections

### ✅ What Surprised Me:
- DistilBERT worked well even without fine-tuning
- ResNet image features improved class separation

### ❓ Room for Improvement:
- Add personalization via user embeddings
- Try contrastive models like CLIP for vision-text fusion
- Scale up to full Amazon dataset with distributed training

---

## 🔗 References

- [Chen et al., 2020] Multimodal Recommender Systems
- [McAuley et al., 2015] Image-based Recommendations
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Amazon Review Data](https://nijianmo.github.io/amazon/index.html)
- TensorFlow/Keras Documentation
- ChatGPT

---






