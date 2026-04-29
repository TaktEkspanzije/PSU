'''
Room occupancy classification - KNN
Zadatak 2

R.Grbic, 2024.
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay


df = pd.read_csv('occupancy_processed.csv')

feature_names = ['S3_Temp', 'S5_CO2']
target_name = 'Room_Occupancy_Count'
class_names = ['Slobodna', 'Zauzeta']

X = df[feature_names].to_numpy()
y = df[target_name].to_numpy()


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Skup za ucenje:   {X_train.shape[0]} primjera")
print(f"Skup za testiranje: {X_test.shape[0]} primjera")


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_scaled, y_train)


y_pred = knn.predict(X_test_scaled)


cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues')
plt.title(f'Matrica zabune (KNN, k={k})')
plt.tight_layout()
plt.show()


acc = accuracy_score(y_test, y_pred)
print(f"\nTocnost klasifikacije (k={k}): {acc:.4f} ({acc*100:.2f}%)")


prec = precision_score(y_test, y_pred, average=None)
rec  = recall_score(y_test, y_pred, average=None)
print(f"\n{'Klasa':<12} {'Preciznost':>12} {'Odziv':>10}")
print("-" * 36)
for i, name in enumerate(class_names):
    print(f"{name:<12} {prec[i]:>12.4f} {rec[i]:>10.4f}")


print("\n--- e) Utjecaj broja susjeda na tocnost ---")
print(f"{'k':<6} {'Tocnost':>10}")
print("-" * 18)
for k_val in [1, 3, 5, 10, 20, 50, 100]:
    knn_k = KNeighborsClassifier(n_neighbors=k_val)
    knn_k.fit(X_train_scaled, y_train)
    acc_k = accuracy_score(y_test, knn_k.predict(X_test_scaled))
    print(f"{k_val:<6} {acc_k:.4f}")


print("\n--- f) Bez skaliranja ulaznih velicina (k=5) ---")
knn_no_scale = KNeighborsClassifier(n_neighbors=5)
knn_no_scale.fit(X_train, y_train)
acc_no_scale = accuracy_score(y_test, knn_no_scale.predict(X_test))
print(f"Tocnost BEZ skaliranja: {acc_no_scale:.4f} ({acc_no_scale*100:.2f}%)")
print(f"Tocnost SA skaliranjem: {accuracy_score(y_test, y_pred):.4f} ({accuracy_score(y_test, y_pred)*100:.2f}%)")