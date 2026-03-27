import pandas as pd
import numpy as np

mtcars = pd.read_csv('mtcars.csv')

#1Najveca potrosnja:
print(mtcars.sort_values(by='mpg').tail(5));

#2Koja tri automobila s 8 cilindara imaju najmanju potrošnju?
auto_8cyl = mtcars[mtcars.cyl == 8];

print(auto_8cyl.sort_values(by='mpg').head(3));

#3. Kolika je srednja potrošnja automobila sa 6 cilindara?

auto_6cyl = mtcars[mtcars.cyl == 6];
print(auto_6cyl['mpg'].mean);

#4. Kolika je srednja potrošnja automobila s 4 cilindra mase između 2000 i 2200 lbs?


auto_4cyl = mtcars[(mtcars.cyl == 4) & (mtcars['wt']>2) & (mtcars['wt']<2.2)];
print(auto_4cyl['mpg'].mean);

# 5. Koliko je automobila s ručnim, a koliko s automatskim mjenjačem u ovom skupu podataka?

broj_automackih = (mtcars['am'] == 1).sum();
broj_rucnih = (mtcars['am'] == 0).sum();

print('Broj automackih:');   print(broj_automackih       );   print('Broj rucnih:');   print(broj_rucnih);

#6. Koliko je automobila s automatskim mjenjačem i snagom preko 100 konjskih snaga?

snazni_rucni_auti = mtcars[(mtcars['am'] == 0) & (mtcars['hp']>100)];
print(snazni_rucni_auti);