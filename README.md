# Intrusion Detection System using Machine Learning & GAN

##  Project Overview
This project presents an AI-based Intrusion Detection System (IDS) designed to classify network traffic as **normal** or **malicious**.  
It leverages machine learning techniques and enhances detection capability using **Generative Adversarial Networks (GANs)** to generate synthetic attack data.

---

##  Problem Statement
With the rapid evolution of cyber threats, traditional IDS systems struggle to detect **unknown or modified attacks**.  
This project aims to improve detection accuracy by combining:
- Machine Learning classification
- Synthetic data generation using GANs

---

##  Technologies Used
- Python  
- Pandas & NumPy  
- Scikit-learn  
- TensorFlow / Keras  
- Random Forest Classifier  

---

## Dataset
The model is trained and tested using the **NSL-KDD dataset**:
- `KDDTrain+.txt` → Training dataset  
- `KDDTest+.txt` → Testing dataset  

---

##  Methodology
1. **Data Preprocessing**
   - Handling categorical features
   - Label encoding
   - Feature scaling

2. **Model Training**
   - Random Forest classifier for intrusion detection

3. **GAN-based Enhancement**
   - Generate synthetic attack samples
   - Improve model robustness against unseen attacks

4. **Evaluation**
   - Accuracy Score
   - Classification Report

---

## Key Features
✔ Detects normal vs malicious traffic  
✔ Handles categorical network features  
✔ Improves detection using synthetic data  
✔ Scalable for real-world IDS systems  

---

##  Project Structure
IDS_Project/
│
├── data/
│ ├── KDDTrain+.txt
│ ├── KDDTest+.txt
│
├── ids_model.py
├── README.md

## Future Enhancements
Implement advanced GANs (WGAN, CTGAN)
Multi-class attack classification
Real-time intrusion detection system
Deployment using web interface

## Author
Lakshana G
