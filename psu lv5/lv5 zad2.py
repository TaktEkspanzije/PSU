import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, 
    precision_score, recall_score, accuracy_score, classification_report
)

df = pd.read_csv("occupancy_processed.csv")

features = ['S3_Temp', 'S5_CO2']
target = 'Room_Occupancy_Count'

X = df[features].values
y = df[target].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

matrix = confusion_matrix(y_test, y_pred)
print("Matrica zabune:\n", matrix)

accuracy = accuracy_score(y_test, y_pred)
print("Točnost modela:", accuracy)

print("Izvještaj klasifikacije:\n", classification_report(y_test, y_pred, target_names=['Slobodna', 'Zauzeta']))

disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=['Free', 'Occupied'])
disp.plot(cmap=plt.cm.Reds)
plt.title('Confusion matrix')
plt.show()

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Preciznost: {precision:.3f}")
print(f"Odziv: {recall:.3f}")

"""
Izgradite i evaluirajte algoritam K najbližih susjeda. Slijedite ovaj redoslijed:
a) Podijelite podatke na skup za učenje i skup za testiranje (omjer 80%-20%) pomoću funkcije
train_test_split. Koristite opciju stratify=y.
b) Pomoću StandardScaler skalirajte ulazne veličine.
c) Pomoću klase KNeighborsClassifier izgradite algoritam K najbližih susjeda.
d) Evaluirajte izgrađeni klasifikator na testnom skupu podataka:
a. prikažite matricu zabune
b. izračunajte točnost klasifikacije
c. izračunajte preciznost i odziv po klasama
e) Što se događa s rezultatima ako se koristi veći odnosno manji broj susjeda?
Rezultati se mogu ili smanjiti ili povećati ovisno o klasi susjeda.
f) Što se događa s rezultatima ako ne koristite skaliranje ulaznih veličina?
Točnost opada za 2% tj 0.02.
Preciznost i odaziv rastu.
"""