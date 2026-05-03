import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

import tensorflow as tf
Sequential = tf.keras.models.Sequential
Dense = tf.keras.layers.Dense
# -------------------------------
# 1. Load dataset
# -------------------------------
train = pd.read_csv("data/KDDTrain+.txt", header=None)
test = pd.read_csv("data/KDDTest+.txt", header=None)

# -------------------------------
# 2. Column names
# -------------------------------
columns = [
"duration","protocol_type","service","flag","src_bytes","dst_bytes",
"land","wrong_fragment","urgent","hot","num_failed_logins",
"logged_in","num_compromised","root_shell","su_attempted","num_root",
"num_file_creations","num_shells","num_access_files","num_outbound_cmds",
"is_host_login","is_guest_login","count","srv_count","serror_rate",
"srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
"diff_srv_rate","srv_diff_host_rate","dst_host_count",
"dst_host_srv_count","dst_host_same_srv_rate",
"dst_host_diff_srv_rate","dst_host_same_src_port_rate",
"dst_host_srv_diff_host_rate","dst_host_serror_rate",
"dst_host_srv_serror_rate","dst_host_rerror_rate",
"dst_host_srv_rerror_rate","label","difficulty"
]

train.columns = columns
test.columns = columns

train = train.drop("difficulty", axis=1)
test = test.drop("difficulty", axis=1)

# -------------------------------
# 3. Encode categorical features
# -------------------------------
cat_cols = ["protocol_type", "service", "flag"]

for col in cat_cols:
    le = LabelEncoder()
    le.fit(train[col])  # avoid data leakeage
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])

# -------------------------------
# 4. Convert labels
# -------------------------------
train['label'] = train['label'].apply(lambda x: 0 if x == 'normal' else 1)
test['label'] = test['label'].apply(lambda x: 0 if x == 'normal' else 1)

# -------------------------------
# 5. Feature Engineering
# -------------------------------
train['perturbation_score'] = np.abs(train['src_bytes'] - train['dst_bytes']) / (train['src_bytes'] + 1)
test['perturbation_score'] = np.abs(test['src_bytes'] - test['dst_bytes']) / (test['src_bytes'] + 1)

train['traffic_ratio'] = train['count'] / (train['srv_count'] + 1)
test['traffic_ratio'] = test['count'] / (test['srv_count'] + 1)

# -------------------------------
# 6. Split data
# -------------------------------
X_train = train.drop('label', axis=1)
y_train = train['label']

X_test = test.drop('label', axis=1)
y_test = test['label']

# -------------------------------
# 7. Normalize
# -------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# 8. Handle imbalance (SMOTE)
# -------------------------------
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# -------------------------------
# 9. GAN - Train on attack data
# -------------------------------
X_attack = X_train_res[y_train_res == 1].astype(np.float32)

latent_dim = 32
input_dim = X_attack.shape[1]

# Generator
generator = Sequential([
    Dense(64, activation='relu', input_dim=latent_dim),
    Dense(128, activation='relu'),
    Dense(input_dim, activation='tanh')
])

# Discriminator
discriminator = Sequential([
    Dense(128, activation='relu', input_dim=input_dim),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

discriminator.compile(optimizer='adam', loss='binary_crossentropy')

# GAN model
discriminator.trainable = False
gan = Sequential([generator, discriminator])
gan.compile(optimizer='adam', loss='binary_crossentropy')

# Train GAN
epochs = 500
batch_size = 64

for epoch in range(epochs):
    idx = np.random.randint(0, X_attack.shape[0], batch_size)
    real_samples = X_attack[idx]

    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    fake_samples = generator.predict(noise, verbose=0)

    discriminator.train_on_batch(real_samples, np.ones((batch_size, 1)))
    discriminator.train_on_batch(fake_samples, np.zeros((batch_size, 1)))

    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    gan.train_on_batch(noise, np.ones((batch_size, 1)))

    if epoch % 100 == 0:
        print(f"GAN Training Epoch: {epoch}")

# -------------------------------
# 10. Generate synthetic attacks
# -------------------------------
num_fake = 3000
noise = np.random.normal(0, 1, (num_fake, latent_dim))
X_fake = generator.predict(noise)
y_fake = np.ones(num_fake)

# -------------------------------
# 11. Combine real + fake data
# -------------------------------
X_train_final = np.vstack((X_train_res, X_fake))
y_train_final = np.hstack((y_train_res, y_fake))

# -------------------------------
# 12. Train Random Forest
# -------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_final, y_train_final)

# -------------------------------
# 13. Prediction + Uncertainty
# -------------------------------
y_pred = model.predict(X_test)

probs = model.predict_proba(X_test)
uncertainty = 1 - np.max(probs, axis=1)

# -------------------------------
# 14. Evaluation
# -------------------------------
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

print("\nSample Uncertainty Scores:")
print(uncertainty[:10])