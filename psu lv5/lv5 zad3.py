'''
Room occupancy classification - Stablo odlucivanja
Zadatak 3

R.Grbic, 2024.
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay

# --- Ucitavanje podataka ---
df = pd.read_csv('occupancy_processed.csv')

feature_names = ['S3_Temp', 'S5_CO2']
target_name = 'Room_Occupancy_Count'
class_names = ['Slobodna', 'Zauzeta']

X = df[feature_names].to_numpy()
y = df[target_name].to_numpy()

# --- a) Podjela na skup za ucenje i testiranje (80/20, stratificirano) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Skup za ucenje:     {X_train.shape[0]} primjera")
print(f"Skup za testiranje: {X_test.shape[0]} primjera")

# --- b) Skaliranje ulaznih velicina ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# --- c) Izgradnja stabla odlucivanja ---
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train_scaled, y_train)

# --- d) Evaluacija na testnom skupu ---
y_pred = dt.predict(X_test_scaled)

# a. Matrica zabune
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues')
plt.title('Matrica zabune (Stablo odlucivanja, max_depth=3)')
plt.tight_layout()
plt.show()

# b. Tocnost
acc = accuracy_score(y_test, y_pred)
print(f"\nTocnost klasifikacije (max_depth=3): {acc:.4f} ({acc*100:.2f}%)")

# c. Preciznost i odziv po klasama
prec = precision_score(y_test, y_pred, average=None)
rec  = recall_score(y_test, y_pred, average=None)
print(f"\n{'Klasa':<12} {'Preciznost':>12} {'Odziv':>10}")
print("-" * 36)
for i, name in enumerate(class_names):
    print(f"{name:<12} {prec[i]:>12.4f} {rec[i]:>10.4f}")

# --- Vizualizacija stabla ---
plt.figure(figsize=(14, 6))
plot_tree(
    dt,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,
    rounded=True,
    fontsize=11
)
plt.title('Stablo odlucivanja (max_depth=3)')
plt.tight_layout()
plt.show()

# --- e) Utjecaj parametra max_depth ---
print("\n--- Utjecaj parametra max_depth na tocnost ---")
print(f"{'max_depth':<12} {'Tocnost':>10} {'Broj listova':>14}")
print("-" * 38)
for depth in [1, 2, 3, 4, 5, 10, None]:
    dt_d = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt_d.fit(X_train_scaled, y_train)
    acc_d = accuracy_score(y_test, dt_d.predict(X_test_scaled))
    leaves = dt_d.get_n_leaves()
    depth_label = str(depth) if depth is not None else "None (potpuno)"
    print(f"{depth_label:<12} {acc_d:.4f}     {leaves:>6}")

# --- f) Bez skaliranja ---
print("\n--- Bez skaliranja ulaznih velicina (max_depth=3) ---")
dt_no_scale = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_no_scale.fit(X_train, y_train)
acc_no_scale = accuracy_score(y_test, dt_no_scale.predict(X_test))
print(f"Tocnost BEZ skaliranja: {acc_no_scale:.4f} ({acc_no_scale*100:.2f}%)")
print(f"Tocnost SA skaliranjem: {acc:.4f} ({acc*100:.2f}%)")
print("(Stablo odlucivanja ne ovisi o skaliranju - rezultati su jednaki)")