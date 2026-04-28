# 🛡️ Intrusion Detection System using Machine Learning & GAN

## 📌 Project Description
This project is an AI-based Intrusion Detection System (IDS) that detects malicious and normal network traffic using machine learning techniques. It is further enhanced using a GAN-based approach to generate synthetic attack data for improved detection of unseen attacks.

---

## 🎯 Problem Statement
Cyber attacks are evolving rapidly, and traditional IDS systems fail to detect new or modified attack patterns. This project aims to improve detection accuracy using machine learning and adversarial data generation.

---

## ⚙️ Technologies Used
- Python
- Pandas
- Scikit-learn
- TensorFlow (for GAN)
- Random Forest Classifier

---

## 📊 Dataset
:contentReference[oaicite:1]{index=1}  
- KDDTrain+.txt → Training data  
- KDDTest+.txt → Testing data  

---

## 🧠 Approach
1. Data preprocessing (encoding, cleaning, scaling)
2. Train Random Forest classifier for IDS
3. Generate synthetic attack data using GAN
4. Improve model training with augmented data
5. Evaluate performance using accuracy and classification report

---

## 🚀 Features
- Detects normal vs attack traffic
- Handles categorical network features
- Machine learning-based classification
- GAN-based data augmentation (enhancement stage)

---

## 📁 Project Structure
IDS_Project/
│
├── data/
│ ├── KDDTrain+.txt
│ ├── KDDTest+.txt
│
├── ids_model.py
├── README.md


---

## 📈 Future Improvements
- Improve GAN model (WGAN / CTGAN)
- Multi-class attack classification
- Real-time intrusion detection system

---

## 👩‍💻 Author
Lakshana
