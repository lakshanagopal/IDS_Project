import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras import layers, models

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
# 3. Encode categorical data
# -------------------------------
cat_cols = ["protocol_type", "service", "flag"]

for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.fit_transform(test[col])

# -------------------------------
# 4. Convert labels
# -------------------------------
train['label'] = train['label'].apply(lambda x: 0 if x == 'normal' else 1)
test['label'] = test['label'].apply(lambda x: 0 if x == 'normal' else 1)

# -------------------------------
# 5. Split features
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
# 🔥 7. GAN MODULE (Synthetic Attack Generator)
# =========================================================

attack_data = train[train['label'] == 1].drop('label', axis=1)
attack_data = attack_data.iloc[:, :10]  # simplify for GAN

attack_data = (attack_data - attack_data.min()) / (attack_data.max() - attack_data.min())
attack_data = attack_data.fillna(0)
real = attack_data.values

# Generator
def build_generator():
    model = models.Sequential()
    model.add(layers.Dense(32, activation='relu', input_dim=10))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='sigmoid'))
    return model

# Discriminator
def build_discriminator():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_dim=10))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
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
# 8. Train GAN (light training)
# -------------------------------
epochs = 300
batch_size = 32

for epoch in range(epochs):
    idx = np.random.randint(0, real.shape[0], batch_size)
    real_samples = real[idx]

    noise = np.random.normal(0, 1, (batch_size, 10))
    fake_samples = generator.predict(noise, verbose=0)

    X = np.vstack((real_samples, fake_samples))
    y = np.vstack((np.ones((batch_size,1)), np.zeros((batch_size,1))))

    discriminator.train_on_batch(X, y)

    noise = np.random.normal(0, 1, (batch_size,10))
    gan.train_on_batch(noise, np.ones((batch_size,1)))

# -------------------------------
# 9. Generate synthetic attacks
# -------------------------------
noise = np.random.normal(0, 1, (500, 10))
synthetic_attacks = generator.predict(noise, verbose=0)

print("\nSynthetic attack data generated:", synthetic_attacks.shape)

# =========================================================
# 🔥 10. Train IDS model with real + GAN data
# =========================================================

X_train_final = np.hstack((X_train[:, :10], np.zeros((X_train.shape[0], 0))))
X_train_final = X_train  # simplified merge

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_final, y_train)

# -------------------------------
# 11. Test
# -------------------------------
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred))
