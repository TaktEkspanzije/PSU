import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ucitavanje ociscenih podataka
df = pd.read_csv('cars_processed.csv')
print(df.info())

# razliciti prikazi
sns.pairplot(df, hue='fuel')

#auto s najvecom i najmanjom cijenom
auto_max_cijena = df.loc[df['selling_price'].idxmax()]
auto_min_cijena = df.loc[df['selling_price'].idxmin()]
print(f"Najveca cijena: {auto_max_cijena['name']} ({auto_max_cijena['selling_price']})")
print(f"Najmanja cijena: {auto_min_cijena['name']} ({auto_min_cijena['selling_price']})")

#broj automobila proizvedenih 2012. godine
godina_N_proizvodnje = len(df[df['year'] == 2012])
print(f"Broj automobila proizvedenih 2012. godine: {godina_N_proizvodnje}")

#automobil s najvecom i najmanjom kilometrazom
max_km_auto = df.loc[df['km_driven'].idxmax()]
min_km_auto = df.loc[df['km_driven'].idxmin()]
print(f"Najvise kilometara: {max_km_auto['name']} - {max_km_auto['km_driven']} km")
print(f"Najmanje kilometara: {min_km_auto['name']} - {min_km_auto['km_driven']} km")

#najcesci broj sjedala i broj automobila s tim brojem sjedala
najcesci_broj_sjedala = df['seats'].mode()[0]
broj_automobila = len(df[df['seats'] == najcesci_broj_sjedala])
print(f"Automobili najcesce imaju {najcesci_broj_sjedala} sjedala.")
print(f"U skupu podataka ima {broj_automobila} takvih automobila.")

#prosjecna kilometraza za automobile s dizel motorom i benzinskim motorom
prosjek_po_gorivu = df.groupby('fuel')['km_driven'].mean()
print("Prosjecna kilometraza prema vrsti goriva:")
print(f"Diesel: {prosjek_po_gorivu['Diesel']:.2f} km")
print(f"Petrol: {prosjek_po_gorivu['Petrol']:.2f} km")



sns.relplot(data=df, x='km_driven', y='selling_price', hue='fuel')
df = df.drop(['name','mileage'], axis=1)

obj_cols = df.select_dtypes(object).columns.values.tolist()
num_cols = df.select_dtypes(np.number).columns.values.tolist()

fig = plt.figure(figsize=[15,8])
for col in range(len(obj_cols)):
    plt.subplot(2,2,col+1)
    sns.countplot(x=obj_cols[col], data=df)
plt.tight_layout()
df.boxplot(by ='fuel', column =['selling_price'], grid = False)

df.hist(['selling_price'], grid = False)

numeric_df = df.select_dtypes(include=[np.number])
tabcorr = numeric_df.corr()
sns.heatmap(tabcorr, annot=True, linewidths=2, cmap= 'coolwarm') 

plt.show()


"""
1. 
6699

2. 
name - object
year - int64
selling_price - float64
km_driven - int64
fuel - object
seller_type - object
transmission - object
owner - object
mileage - float64
engine - int64
max_power - float64
seats - int64

3. 
Najveća cijena: BMW X7 xDrive 30d DPE (15.789591583986285)
Najmanja cijena: Maruti 800 AC (10.308919326755392)

4. 
575 automobila je proizvedeno 2012. godine.

5. 
Najvise kilometara: Maruti Wagon R LXI Minor - 577414 km
Najmanje kilometara: Maruti Eeco 5 STR With AC Plus HTR CNG - 1 km

6. 
Automobili najcesce imaju 5 sjedala.

7. 
Prosjecna kilometraza prema vrsti goriva:
Diesel: 88039.97 km
Petrol: 54101.88 km
"""