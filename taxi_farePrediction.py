# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 21:30:16 2019

@author: tauab
"""

import pandas as pd
import numpy as np

n = 55000000 #Tamanho aproximado de linhas fornecido no desafio
s = 100000 # 
skip = sorted(np.random.choice(range(n), n-s, replace=False))
skip[0] = 1
df = pd.read_csv('train.csv', skiprows=skip, header=0)

df.describe()

from math import radians, cos, sin, asin, sqrt

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371
    return c * r

listaTempo = []
listaMes = []
listaAno = []
listaDia = []
listaNomeDia = []
listaDistancia = []

df['pickup_datetime']=pd.to_datetime(df['pickup_datetime'],format='%Y-%m-%d %H:%M:%S UTC')

for index, row in df.iterrows():
    listaMes.append(row['pickup_datetime'].month)
    listaAno.append(row['pickup_datetime'].year)
    listaDistancia.append(haversine(row['pickup_longitude'], row['pickup_latitude'], row['dropoff_longitude'], row['dropoff_latitude']))

df['distance'] = listaDistancia
df['mes'] = listaMes
df['ano'] = listaAno
df.describe()

#Filtrando os valores errados nas coordenadas
df.drop(df[df['dropoff_latitude'] < 30].index, inplace = True)
df.drop(df[df['pickup_latitude'] < 30].index, inplace = True)
df.drop(df[df['pickup_longitude'] > -60].index, inplace = True)
df.drop(df[df['dropoff_longitude'] > -60].index, inplace = True)

df.drop(df[df['dropoff_latitude'] > 50].index, inplace = True)
df.drop(df[df['pickup_latitude'] > 50].index, inplace = True)
df.drop(df[df['pickup_longitude'] < -80].index, inplace = True)
df.drop(df[df['dropoff_longitude'] < -80].index, inplace = True)

df.drop(df[df['distance'] < 10**-1].index, inplace = True)
df.drop(df[df['distance'] > 180].index, inplace = True)

#Filtrando os valores errados no pagamento
df.drop(df[df['fare_amount'] < 1].index, inplace = True)

#Filtrando os valores errados no Número de passageiros (no caso, como os dados que devemos prever só vão até 6 passageiros,
#e um carro grande geralmente cabe no máximo 6 pessoas, optei por usar 6 como o número máximo de pessoas)
df.drop(df[df['passenger_count'] == 0].index, inplace = True)
df.drop(df[df['passenger_count'] > 6].index, inplace = True)
df.drop(df[(df['passenger_count']*10)%10 > 0].index, inplace = True)

df.describe()

import numpy as np

var1 = np.array(df[df['ano'] == 2012]['fare_amount'])
var2 = np.array(df[df['ano'] == 2012]['distance'])
var1log = np.log10( np.array(df[df['ano'] == 2012]['fare_amount']))
var2log = np.log10( np.array(df[df['ano'] == 2012]['distance']))
np.corrcoef(var1, var2)


var3 = np.array(df[df['mes'] == 12]['fare_amount'])
var4 = np.array(df[df['mes'] == 12]['distance'])
var3log = np.log10( np.array(df[df['mes'] == 12]['fare_amount']))
var4log = np.log10( np.array(df[df['mes'] == 12]['distance']))
np.corrcoef(var3, var4)

import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
matplotlib.style.use('ggplot')
plt.xlabel("Log(Distancia)")
plt.ylabel("Log(Preço)")
plt.title("Distancia vs Preço no ano de 2012")
plt.scatter(var2log, var1log) 
plt.show()
plt.xlabel("Log(Distancia)")
plt.ylabel("Log(Preço)")
plt.title("Distancia vs Preço no mês de Dezembro")
plt.scatter(var3log, var4log) 
plt.show()

a = df.drop(['Unnamed: 0', 'key', 'pickup_datetime', 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude' ], axis = 1)

labels = np.array(a['fare_amount'])
a.drop('fare_amount', axis = 1, inplace = True)
feature_list = list(a.columns)
a = np.array(a)
#df2 = pd.DataFrame(df2)
from sklearn.model_selection import train_test_split

train_features, test_features, train_labels, test_labels = train_test_split(a, labels, test_size = 0.25, random_state = 42)

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators = 500, random_state = 42, max_depth = 24, max_features = 4)

rf.fit(train_features, train_labels);

predictions = rf.predict(test_features)

def rmse(predictions, targets):

    differences = predictions - targets      

    differences_squared = differences ** 2                 

    mean_of_differences_squared = differences_squared.mean()

    rmse_val = np.sqrt(mean_of_differences_squared)          

    return rmse_val  
  
error = rmse(predictions,test_labels)
print('Mean Absolute Error:', error, 'USD')