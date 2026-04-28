import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

import tensorflow as tf
from keras import layers, models

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
# 3. Encode categorical
# -------------------------------
cat_cols = ["protocol_type", "service", "flag"]

for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])

# -------------------------------
# 4. Labels
# -------------------------------
train['label'] = train['label'].apply(lambda x: 0 if x == 'normal' else 1)
test['label'] = test['label'].apply(lambda x: 0 if x == 'normal' else 1)

# -------------------------------
# 5. Split
# -------------------------------
X_train = train.drop('label', axis=1)
y_train = train['label']

X_test = test.drop('label', axis=1)
y_test = test['label']

# -------------------------------
# 6. Normalize
# -------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================================================
#  GAN (Improved usage)
# =========================================================

attack_data = train[train['label'] == 1].drop('label', axis=1)
attack_data = attack_data.iloc[:, :10]

attack_data = (attack_data - attack_data.min()) / (attack_data.max() - attack_data.min())
attack_data = attack_data.fillna(0)

real = attack_data.values

def build_generator():
    model = models.Sequential([
        layers.Dense(32, activation='relu', input_dim=10),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='sigmoid')
    ])
    return model

def build_discriminator():
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_dim=10),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

generator = build_generator()
discriminator = build_discriminator()

discriminator.compile(optimizer='adam', loss='binary_crossentropy')

discriminator.trainable = False

gan_input = layers.Input(shape=(10,))
generated = generator(gan_input)
output = discriminator(generated)

gan = models.Model(gan_input, output)
gan.compile(optimizer='adam', loss='binary_crossentropy')

# -------------------------------
# Train GAN
# -------------------------------
for epoch in range(200):
    idx = np.random.randint(0, real.shape[0], 32)
    real_samples = real[idx]

    noise = np.random.normal(0, 1, (32, 10))
    fake_samples = generator.predict(noise, verbose=0)

    X = np.vstack((real_samples, fake_samples))
    y = np.vstack((np.ones((32,1)), np.zeros((32,1))))

    discriminator.train_on_batch(X, y)

    noise = np.random.normal(0, 1, (32,10))
    gan.train_on_batch(noise, np.ones((32,1)))

# -------------------------------
# Generate synthetic attacks
# -------------------------------
noise = np.random.normal(0, 1, (500, 10))
synthetic_attacks = generator.predict(noise, verbose=0)

# -------------------------------
#  Merge GAN data (IMPORTANT)
# -------------------------------
synthetic_full = np.zeros((synthetic_attacks.shape[0], X_train.shape[1]))
synthetic_full[:, :10] = synthetic_attacks

y_synthetic = np.ones(synthetic_full.shape[0])

X_train_final = np.vstack((X_train, synthetic_full))
y_train_final = np.hstack((y_train, y_synthetic))

# -------------------------------
# Train model
# -------------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_final, y_train_final)

# -------------------------------
# Test
# -------------------------------
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred))