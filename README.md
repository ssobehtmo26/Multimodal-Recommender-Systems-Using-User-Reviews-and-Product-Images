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
| ✅ Accuracy         | **~61.3%**     |
| 📊 Best Class       | 5-star ratings |
| 🧪 Observations     | Better performance on 4-5 stars; difficult to differentiate mid-tier (3-star) |

![Screenshot 2025-05-07 140211](https://github.com/user-attachments/assets/26eb18cc-7371-4e9c-9a9f-b9aeed234a9a)

![Screenshot 2025-05-07 201654](https://github.com/user-attachments/assets/8dceedbe-a88a-4d01-900b-3f95bee05304)

![Screenshot 2025-05-07 201610](https://github.com/user-attachments/assets/42e01c80-95a3-4a22-b734-058a5af5ba9c)

![Screenshot 2025-05-07 201620](https://github.com/user-attachments/assets/0a4e8fd5-5687-4565-986e-90881acadce6)

![Screenshot 2025-05-07 201632](https://github.com/user-attachments/assets/e5653a42-a2cc-420f-8a8e-d91383fa8927)

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






