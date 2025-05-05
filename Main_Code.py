# 1. Importowanie niezbędnych bibliotek
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Funkcje i klasy Scikit-learn
from sklearn.datasets import load_iris # Do załadowania zbioru Iris
from sklearn.model_selection import train_test_split # Do podziału danych na zbiór treningowy i testowy
from sklearn.preprocessing import StandardScaler # Do skalowania cech (ważne dla KNN)
from sklearn.neighbors import KNeighborsClassifier # Klasyfikator K-Najbliższych Sąsiadów
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score # Metryki oceny modelu

print("Biblioteki zaimportowane pomyślnie.")

# 2. Ładowanie danych Iris
# Używamy as_frame=True, aby od razu otrzymać dane w formacie Pandas DataFrame
iris = load_iris(as_frame=True)
X = iris.data # Cechy (wymiary kwiatów) - DataFrame
y = iris.target # Etykiety (gatunki kwiatów) - Series
feature_names = iris.feature_names # Nazwy cech
target_names = iris.target_names # Nazwy gatunków

print("\n--- Wczytane dane ---")
print("Nazwy cech:", feature_names)
print("Nazwy gatunków (klas):", target_names)
print("Pierwsze 5 wierszy cech (X):")
print(X.head())
print("\nPierwsze 5 etykiet (y):")
print(y.head())
print(f"\nRozmiar danych: {X.shape[0]} próbek, {X.shape[1]} cech.")

# 3. Podział danych na zbiór treningowy i testowy
# Dzielimy dane, aby móc ocenić model na danych, których nie widział podczas treningu.
# test_size=0.3 oznacza, że 30% danych trafi do zbioru testowego.
# random_state zapewnia powtarzalność podziału - przy każdym uruchomieniu podział będzie taki sam.
# stratify=y zapewnia, że proporcje klas w zbiorze treningowym i testowym będą takie same jak w oryginalnym zbiorze.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print("\n--- Podział danych ---")
print(f"Rozmiar zbioru treningowego (X_train): {X_train.shape}")
print(f"Rozmiar zbioru testowego (X_test): {X_test.shape}")
print(f"Rozkład klas w zbiorze treningowym:\n{y_train.value_counts(normalize=True)}")
print(f"Rozkład klas w zbiorze testowym:\n{y_test.value_counts(normalize=True)}")

# 4. Skalowanie cech
# Algorytm KNN jest wrażliwy na skalę cech (ponieważ bazuje na odległościach).
# Cechy o większych wartościach mogłyby zdominować obliczenia.
# Używamy StandardScaler, który przekształca dane tak, aby miały średnią 0 i odchylenie standardowe 1.
# WAŻNE: Skaler 'uczymy' (fit) tylko na danych treningowych, a następnie transformujemy oba zbiory (treningowy i testowy).
# Zapobiega to "wyciekowi" informacji ze zbioru testowego do procesu treningowego.

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # Uczymy i transformujemy zbiór treningowy
X_test_scaled = scaler.transform(X_test) # Transformujemy zbiór testowy używając dopasowanego skalera

print("\n--- Skalowanie cech zakończone ---")
# Można opcjonalnie wydrukować np. średnią i odch. std. po skalowaniu dla X_train_scaled, aby zobaczyć efekt
# print("Średnia cech w X_train po skalowaniu:", X_train_scaled.mean(axis=0))
# print("Odch. std. cech w X_train po skalowaniu:", X_train_scaled.std(axis=0))

# 5. Budowa i trenowanie modelu K-Najbliższych Sąsiadów (KNN)
# Wybieramy liczbę sąsiadów (k). To hiperparametr, który można dostrajać. Zacznijmy od k=5.
k = 5
knn = KNeighborsClassifier(n_neighbors=k)

# Trenujemy model na przeskalowanych danych treningowych
knn.fit(X_train_scaled, y_train)

print(f"\n--- Model KNN (k={k}) wytrenowany ---")

# 6. Przewidywanie na danych testowych
# Używamy wytrenowanego modelu do przewidzenia gatunków dla danych testowych.
y_pred = knn.predict(X_test_scaled)

print("\n--- Dokonano predykcji na zbiorze testowym ---")
# Porównanie kilku pierwszych prawdziwych etykiet z przewidzianymi
print("Prawdziwe etykiety (y_test) [pierwsze 10]:", y_test.values[:10])
print("Przewidziane etykiety (y_pred) [pierwsze 10]:", y_pred[:10])

# 7. Ocena modelu

# a) Dokładność (Accuracy)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n--- Ocena modelu ---")
print(f"Dokładność (Accuracy): {accuracy:.4f} (czyli {accuracy*100:.2f}%)")

# b) Raport klasyfikacji (Precision, Recall, F1-score)
# Precision: Jaki procent próbek zaklasyfikowanych jako dana klasa faktycznie nią jest?
# Recall: Jaki procent próbek z danej klasy model poprawnie zidentyfikował?
# F1-score: Średnia harmoniczna precyzji i czułości.
print("\nRaport klasyfikacji:")
# Używamy target_names, aby raport był czytelniejszy
print(classification_report(y_test, y_pred, target_names=target_names))

# c) Macierz pomyłek (Confusion Matrix)
# Pokazuje, ile próbek z każdej prawdziwej klasy zostało zaklasyfikowanych do każdej z możliwych klas.
print("\nMacierz pomyłek:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Wizualizacja macierzy pomyłek
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title('Macierz pomyłek dla modelu KNN (k=5)')
plt.xlabel('Przewidziana klasa')
plt.ylabel('Prawdziwa klasa')
plt.show()

# 8. Podsumowanie i wnioski
print("\n--- Podsumowanie ---")
print(f"Model K-Najbliższych Sąsiadów (z k={k}) został zbudowany i oceniony na zbiorze Iris.")
print(f"Osiągnął dokładność na poziomie {accuracy*100:.2f}% na danych testowych.")
print("Z raportu klasyfikacji i macierzy pomyłek wynika, że model radzi sobie bardzo dobrze z rozróżnianiem gatunków irysów.")
# Można dodać bardziej szczegółowe wnioski, np. czy były jakieś pomyłki i między którymi klasami (na podstawie macierzy pomyłek).
# W tym przypadku model KNN na Iris często osiąga 100% lub blisko 100% dokładności, zwłaszcza po skalowaniu.
num_errors = (y_test != y_pred).sum()
if num_errors == 0:
    print("Model poprawnie zaklasyfikował wszystkie próbki testowe.")
else:
    print(f"Model popełnił {num_errors} błędów na zbiorze testowym.")
    # Tutaj można by dodać analizę błędów, np. które klasy były mylone.

print("\nProjekt zakończony.")