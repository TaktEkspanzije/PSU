import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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

print(f"Skup za ucenje:     {X_train.shape[0]} primjera")
print(f"Skup za testiranje: {X_test.shape[0]} primjera")


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


lr = LogisticRegression(random_state=42)
lr.fit(X_train_scaled, y_train)


y_pred = lr.predict(X_test_scaled)


cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues')
plt.title('Matrica zabune (Logisticka regresija)')
plt.tight_layout()
plt.show()


acc = accuracy_score(y_test, y_pred)
print(f"\nTocnost klasifikacije: {acc:.4f} ({acc*100:.2f}%)")


prec = precision_score(y_test, y_pred, average=None)
rec  = recall_score(y_test, y_pred, average=None)
print(f"\n{'Klasa':<12} {'Preciznost':>12} {'Odziv':>10}")
print("-" * 36)
for i, name in enumerate(class_names):
    print(f"{name:<12} {prec[i]:>12.4f} {rec[i]:>10.4f}")


print("\n--- Razdioba klasa u testnom skupu ---")
unique, counts = np.unique(y_test, return_counts=True)
for cls, cnt in zip(unique, counts):
    print(f"  Klasa {cls} ({class_names[cls]}): {cnt} primjera ({cnt/len(y_test)*100:.1f}%)")

print("\n--- Razdioba predikcija ---")
unique_p, counts_p = np.unique(y_pred, return_counts=True)
for cls, cnt in zip(unique_p, counts_p):
    print(f"  Klasa {cls} ({class_names[cls]}): {cnt} predikcija ({cnt/len(y_pred)*100:.1f}%)")