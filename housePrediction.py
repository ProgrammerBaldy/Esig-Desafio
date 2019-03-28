# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 17:27:17 2019

@author: tauab
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 15:25:17 2019

@author: tauab
"""

import pandas as pd
import numpy as np


df = pd.read_csv('train.csv')
df.describe()

#About the house:
houseStyle, foundation = [], []
totalRoomsAbvGnd, heatQuality, exterior1st, exteriorQuality = [], [], [], []

#About the sale:
saleType, saleCond = [], []

#About the Basement:
bsmtQual, bsmtExposure, bsmtFin1 = [], [], []

#About the Garage:
garageFinished, garageCond, garageQual = [], [], []
#About the kitchen:
kitchenQual = []

for index, row in df.iterrows():
    #House:
    if(row['HouseStyle'] == "2.5Fin" or row['HouseStyle'] == "2Story"):
        houseStyle.append(10)
    elif (row['HouseStyle'] == "1Story" or row['HouseStyle'] == "SLvl"):
        houseStyle.append(7)
    elif (row['HouseStyle'] == "1.5Unf"):
        houseStyle.append(3)
    else: houseStyle.append(5)
    
    if(row['Foundation'] == "PConc" or row['Foundation'] == "Wood"):
        foundation.append(10)
    elif (row['Foundation'] == "Stone" or row['Foundation'] == "CBlock"):
        foundation.append(6)
    else: foundation.append(4)
    
    if(row['TotRmsAbvGrd'] > 9):
        totalRoomsAbvGnd.append(10)
    elif (row['TotRmsAbvGrd'] > 7):
        totalRoomsAbvGnd.append(8)
    elif (row['TotRmsAbvGrd'] > 5):
        totalRoomsAbvGnd.append(6)
    elif (row['TotRmsAbvGrd'] > 2):
        totalRoomsAbvGnd.append(4)
    else: totalRoomsAbvGnd.append(2)
    
    if(row['HeatingQC'] == "Ex"):
        heatQuality.append(10)
    elif (row['HeatingQC'] == "Gd"):
        heatQuality.append(7)
    elif (row['HeatingQC'] == "TA"):
        heatQuality.append(5.5)
    elif (row['HeatingQC'] == "Fa"):
        heatQuality.append(4.5)
    else: heatQuality.append(3)
    
    if(row['Exterior1st'] == "ImStucc" or row['Exterior1st'] == "Stone"):
        exterior1st.append(10)
    elif (row['Exterior1st'] == "CemntBd" or row['Exterior1st'] == "VinylSd" or row['Exterior1st'] == "BrkFace"):
        exterior1st.append(7)
    elif (row['Exterior1st'] == "Plywood" or row['Exterior1st'] == "HdBoard" or row['Exterior1st'] == "Stucco"):
        exterior1st.append(5)
    elif (row['Exterior1st'] == "WdShing" or row['Exterior1st'] == "Wd Sdng" or row['Exterior1st'] == "MetalSd"):    
        exterior1st.append(3)
    else: exterior1st.append(2)
    
    if(row['ExterQual'] == "Ex"):
        exteriorQuality.append(10)
    elif (row['ExterQual'] == "Gd"):
        exteriorQuality.append(7.5)
    elif (row['ExterQual'] == "TA"):
        exteriorQuality.append(5)
    elif (row['ExterQual'] == "Fa"):
        exteriorQuality.append(2.5)
    else: exteriorQuality.append(0)
    
    #Sale:
    if(row['SaleType'] == "New" or row['SaleType'] == "Con"):
        saleType.append(10)
    elif (row['SaleType'] == "CWD" or row['SaleType'] == "ConLi"):
        saleType.append(7.5)
    elif (row['SaleType'] == "WD"):
        saleType.append(6)
    elif (row['SaleType'] == "Oth"):    
        saleType.append(3)
    else: saleType.append(5)
    
    if(row['SaleCondition'] == "Partial"):
        saleCond.append(10)
    elif (row['SaleCondition'] == "Normal" or row['SaleCondition'] == "Alloca"):
        saleCond.append(6.5)
    elif (row['SaleType'] == "AdjLand"):
        saleCond.append(4)
    else: saleCond.append(5)
    
    #Garage:
    if(row['GarageFinish'] == "Fin"):
        garageFinished.append(10)
    elif (row['GarageFinish'] == "RFn"):
        garageFinished.append(8)
    elif (row['GarageFinish'] == "Unf"):
        garageFinished.append(5)
    else: garageFinished.append(0)
    
    if(row['GarageQual'] == "Ex"):
        garageQual.append(10)
    elif (row['GarageQual'] == "Gd"):
        garageQual.append(8)
    elif (row['GarageQual'] == "TA"):
        garageQual.append(6.5)
    elif (row['GarageQual'] == "Fa"):
        garageQual.append(5)
    else: garageQual.append(3)
    
    #Kitchen:
    if(row['KitchenQual'] == "Ex"):
        kitchenQual.append(10)
    elif (row['KitchenQual'] == "Gd"):
        kitchenQual.append(6.5)
    elif (row['KitchenQual'] == "TA"):
        kitchenQual.append(5)
    elif (row['KitchenQual'] == "Fa"):
        kitchenQual.append(3)
    else: kitchenQual.append(0)
    
    #Basement:
    if(row['BsmtQual'] == "Ex"):
        bsmtQual.append(10)
    elif (row['BsmtQual'] == "Gd"):
        bsmtQual.append(6.5)
    elif (row['BsmtQual'] == "TA"):
        bsmtQual.append(5)
    elif (row['BsmtQual'] == "Fa"):
        bsmtQual.append(3)
    else: bsmtQual.append(0)
    
    if(row['BsmtExposure'] == "Gd"):
        bsmtExposure.append(10)
    elif (row['BsmtExposure'] == "Av"):
        bsmtExposure.append(7)
    elif (row['BsmtExposure'] == "Mn"):
        bsmtExposure.append(6)
    elif (row['BsmtExposure'] == "No"):
        bsmtExposure.append(4)
    else: bsmtExposure.append(0)
    
    if(row['BsmtFinType1'] == "GLQ"):
        bsmtFin1.append(10)
    elif (row['BsmtFinType1'] == "Unf" or row['BsmtFinType1'] == "ALQ"):
        bsmtFin1.append(7)
    elif (row['BsmtFinType1'] == "LwQ" or row['BsmtFinType1'] == "BLQ" or row['BsmtFinType1'] == "Rec"):
        bsmtFin1.append(6)
    else: bsmtFin1.append(0)
    
houseMeans = [x + y + z + h + e + e2 for x, y, z, h, e, e2 in zip(houseStyle, foundation, totalRoomsAbvGnd, heatQuality,
                                                     exterior1st, exteriorQuality)]
houseMeans = [x/6 for x in (houseMeans)]

saleMeans = [x + y for x, y in zip(saleType, saleCond)]
saleMeans = [x/2 for x in (saleMeans)]

bsmtMeans = [x + y + z for x, y, z in zip(bsmtQual, bsmtExposure, bsmtFin1)]
bsmtMeans = [x/3 for x in (bsmtMeans)]

garageMeans = [x + y for x, y in zip(garageFinished, garageQual)]
garageMeans = [x/3 for x in (garageMeans)]

kitchenMeans = kitchenQual
#drop columns:

df.drop(['SaleType', 'GarageFinish', 'TotRmsAbvGrd', 'KitchenQual', 'HeatingQC', 'BsmtExposure', 'BsmtFinType1', 'BsmtQual', 'Foundation', 'ExterQual', 'Exterior1st', 'HouseStyle', 'MSSubClass', 'MSZoning', 'Id', 'LotFrontage', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'MasVnrType', 'Exterior2nd', 'RoofMatl', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'ExterCond', 'BsmtCond', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFinType2', 'Heating', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageCars', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleCondition'], axis = 1, inplace = True)

df['GarageMeans'] = garageMeans
df['KitchenMeans'] = kitchenMeans
df['BsmtMeans'] = bsmtMeans
df['SaleMeans'] = saleMeans
df['HouseMeans'] = houseMeans

from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest, f_classif
features = df.drop('SalePrice', axis = 1)
label = df['SalePrice']
features = features.round(2)

features = features.fillna(0)

fit = SelectKBest(f_classif, k="all").fit(features,label)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(features.columns)

 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']
print(featureScores.nlargest(12,'Score')) 

#Pegar os 6 melhores descritores
novoDF = features.drop(['GarageMeans',  'BsmtMeans', 'GarageArea', 'MasVnrArea', 'BsmtUnfSF', 'OverallCond'], axis = 1)


label = np.array(label)
features_numpy = np.array(features)
#df2 = pd.DataFrame(df2)
from sklearn.model_selection import train_test_split

train_features, test_features, train_labels, test_labels = train_test_split(features, label, test_size = 0.25, random_state = 42)

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators = 1000, random_state = 42, max_depth = 24, max_features = 4)

rf.fit(train_features, train_labels);

label.mean()
