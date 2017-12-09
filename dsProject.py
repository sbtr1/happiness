# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 09:47:32 2017

@author: Shawn Ban
"""

#Import packages:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas.rpy.common as com

#Import and tidy data. Ensure csv files in same folder as script:
df2015 = pd.read_csv('2015.csv')
df2016 = pd.read_csv('2016.csv')
df2017 = pd.read_csv('2017.csv')

df2015.drop(df2015.columns[4:], axis=1, inplace=True)
df2015.drop(df2015.columns[2], axis=1, inplace=True)
df2016.drop(df2016.columns[4:], axis=1, inplace=True)
df2016.drop(df2016.columns[1:3], axis=1, inplace=True)
df2017.drop(df2017.columns[3:], axis=1, inplace=True)
df2017.drop(df2017.columns[1], axis=1, inplace=True)

dataset  = pd.merge(df2015, df2016, on='Country', how='inner')
dataset  = pd.merge(dataset, df2017, on='Country', how ='inner')
dataset = dataset.rename(columns={'Happiness Score_x': 'happy2015', 'Happiness Score_y': 'happy2016', 'Happiness.Score': 'happy2017'})

#Visualize the data. Is it reasonable to assume normality?
plt.style.use('ggplot')
fig = plt.figure(figsize=(8,5))
ax1= fig.add_subplot(131)
ax1.hist(dataset.happy2015, color='teal')
ax1.set_title('2015 scores')
ax2= fig.add_subplot(132)
ax2.hist(dataset.happy2016, color='teal')
ax2.set_title('2016 scores')
ax3= fig.add_subplot(133)
ax3.hist(dataset.happy2017, color='teal')
ax3.set_title('2017 scores')


#Let's explore the average happiness score by region:
avg2017 = dataset['happy2017'].groupby(dataset['Region']).mean().sort_values(ascending=True)
avg2016 = dataset['happy2016'].groupby(dataset['Region']).mean().sort_values(ascending=True)
avg2015 = dataset['happy2015'].groupby(dataset['Region']).mean().sort_values(ascending=True)
avg = pd.concat([avg2017,avg2016,avg2015],axis=1)
avg = avg.sort_values(['happy2017'], ascending=True)

dataset['happychg'] = dataset['happy2017'] - dataset['happy2015'] 
plt.style.use('ggplot')
fig = plt.figure(figsize=(8,5))
ax1= fig.add_subplot(111)
ax1.hist(dataset.happychg, color='teal')
ax1.set_title('Change in happiness, 2015 to 2017')

fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(111)
ind = np.arange(10)
rects1 = ax.bar(ind, avg2017, 0.5, color='teal')
plt.xticks(ind)
xtickNames = ax.set_xticklabels(avg2017.index)
plt.setp(xtickNames, rotation=90, fontsize=11)
ax.set_title('Happiness Scores 2017 by Region')

fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(111)
ind = np.arange(10)
rects1 = ax.bar(ind, avg2016, 0.5, color='teal')
plt.xticks(ind)
xtickNames = ax.set_xticklabels(avg2016.index)
plt.setp(xtickNames, rotation=90, fontsize=11)
ax.set_title('Happiness Scores 2016 by Region')

fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(111)
ind = np.arange(10)
rects1 = ax.bar(ind, avg2015, 0.5, color='teal')
plt.xticks(ind)
xtickNames = ax.set_xticklabels(avg2015.index)
plt.setp(xtickNames, rotation=90, fontsize=11)
ax.set_title('Happiness Scores 2015 by Region')

#Let's see how the average score by region varied over time:
y1 = avg['happy2015']  
y2 = avg['happy2016']
y3 = avg['happy2017']
x = np.arange(10)
fig = plt.figure(figsize=(8,5))
ax = plt.subplot(111)
rects1 = ax.bar(x-0.2,y1,width=0.2,color='darkblue',align='center', label='2015')
rects2 = ax.bar(x,y2,width=0.2,color='grey',align='center', label ='2016')
rects3 = ax.bar(x+0.2,y3,width=0.2,color='teal',align='center', label='2017')
ax.set_xticks(x)
xtickNames = ax.set_xticklabels(avg.index)
plt.setp(xtickNames, rotation=90, fontsize=11)
plt.legend()
plt.title('Happiness Scores by Region, 2015 to 2017')


# GDP:
gdp = pd.read_csv('gdppercapita2.csv')
gdp.describe()
gdp.drop(gdp.columns[2:58], axis=1, inplace=True)
gdp = gdp.rename(columns={'Country Name': 'Country', '2014': 'gdp2014', '2015':'gdp2015', '2016':'gdp2016'})

#Missing values
#Drop the row if all observations missing:
#If only 2016 observation missing, we assume 2016 gdp grew at same rate as 2015:
gdp = gdp.dropna(thresh=4)
gdp2015_growth = gdp['gdp2015']/gdp['gdp2014'] - 1
gdp2016_assumed = (1 + gdp2015_growth) * gdp['gdp2015']                    
gdp['gdp2016'] = gdp['gdp2016'].fillna(gdp2016_assumed)                    

dataset  = pd.merge(dataset, gdp, on='Country', how='inner')
#Visualize the data. Not normal, so we do a log transform
fig = plt.figure(figsize=(8,5))
ax1= fig.add_subplot(131)
ax1.hist(dataset.gdp2014, color='lightblue')
ax1.set_title('2014 gdp per capita')
ax2= fig.add_subplot(132)
ax2.hist(dataset.gdp2015, color='lightblue')
ax2.set_title('2015 gdp per capita')
ax3= fig.add_subplot(133)
ax3.hist(dataset.gdp2016, color='lightblue')
ax3.set_title('2016 gdp per capita')


fig = plt.figure(figsize=(8,5))
ax1= fig.add_subplot(131)
ax1.hist(dataset.gdp2014, color='lightblue')
ax1.set_title('2014 gdp per capita')
ax2= fig.add_subplot(132)
ax2.hist(dataset.gdp2015, color='lightblue')
ax2.set_title('2015 gdp per capita')
ax3= fig.add_subplot(133)
ax3.hist(dataset.gdp2016, color='lightblue')
ax3.set_title('2016 gdp per capita')



# Life expectancy:      
lifeexpect = pd.read_csv('life3.csv', encoding='latin1')
lifeexpect.drop(lifeexpect.columns[60], axis=1, inplace=True)
lifeexpect.drop(lifeexpect.columns[2:58], axis=1, inplace=True)
lifeexpect.drop(lifeexpect.columns[0], axis=1, inplace=True)
lifeexpect = lifeexpect.rename(columns={'2014': 'life2014','2015': 'life2015'})
lifeexpect['life2014'] = lifeexpect['life2014'].fillna(lifeexpect['life2014'].mean())
lifeexpect['life2015'] = lifeexpect['life2015'].fillna(lifeexpect['life2015'].mean())    
dataset  = pd.merge(dataset, lifeexpect, on='Country Code', how='inner')

#Infant mortality:
infant = pd.read_csv('infantmort2.csv', encoding='latin1')
infant.drop(infant.columns[61], axis=1, inplace=True)
infant.drop(infant.columns[2:59], axis=1, inplace=True)
infant.drop(infant.columns[0], axis=1, inplace=True)
infant = infant.rename(columns={'2015': 'infant2015','2016': 'infant2016'})
infant['infant2016'] = infant['infant2016'].fillna(infant['infant2016'].mean())   
infant['infant2015'] = infant['infant2015'].fillna(infant['infant2015'].mean())  
dataset  = pd.merge(dataset, infant, on='Country Code', how='inner')

#Press Freedom:
pressfree = pd.read_csv('pressscore.csv', encoding='latin1', decimal=',')
pressfree.drop(pressfree.columns[6:10], axis=1, inplace=True)
pressfree.drop(pressfree.columns[1:5], axis=1, inplace=True)
pressfree.drop(pressfree.columns[3], axis=1, inplace=True)
pressfree = pressfree.rename(columns={'Underlying situation score 2016': 'press2016', 'Score 2015': 'press2015', 'ISO': 'Country Code'})
dataset  = pd.merge(dataset, pressfree, on='Country Code', how='inner')

#Corruption:
corrupt = pd.read_csv('corruption.csv', encoding='latin1', decimal=',')
corrupt.drop(corrupt.columns[0], axis=1, inplace=True)
corrupt = corrupt.rename(columns={'WB Code': 'Country Code', 'CPI2016': 'corrupt2016', 'CPI 2015': 'corrupt2015'})
corrupt['corrupt2015'] = corrupt['corrupt2015'].fillna(corrupt['corrupt2016'])    
dataset  = pd.merge(dataset, corrupt, on='Country Code', how='inner')

#Inequality
gini = pd.read_csv('gini.csv', encoding='latin1')
gini['gini2017'] = gini['gini2017'].fillna(gini['gini2017'].mean()) 
dataset  = pd.merge(dataset, gini, on='Country Code', how='inner')

#Murder
murder = pd.read_csv('murder5.csv', encoding='latin1')
murder['murder2016'] = murder['murder2016'].fillna(murder['murder2016'].mean())
dataset  = pd.merge(dataset, murder, on='Country Code', how='inner')

#dataset.to_csv('data_corr.csv')
datacorr = pd.read_csv('data_corr.csv')
fig = plt.figure(figsize=(10,8))
ax1= fig.add_subplot(241)
ax1.hist(datacorr.happy, color='blue')
ax1.set_title('happiness')
ax2= fig.add_subplot(242)
ax2.hist(datacorr.gdp, color='blue')
ax2.set_title('gdp')
ax3= fig.add_subplot(243)
ax3.hist(datacorr.life_expectancy, color='blue')
ax3.set_title('life_expectancy')
ax4= fig.add_subplot(244)
ax4.hist(datacorr.infant_mortality, color='blue')
ax4.set_title('infant_mortality')
ax5= fig.add_subplot(245)
ax5.hist(datacorr.pressfreedom, color='blue')
ax5.set_title('pressfreedom')
ax6= fig.add_subplot(246)
ax6.hist(datacorr.corruption, color='blue')
ax6.set_title('corruption')
ax7= fig.add_subplot(247)
ax7.hist(datacorr.inequality, color='blue')
ax7.set_title('inequality')
ax8= fig.add_subplot(248)
ax8.hist(datacorr.crime, color='blue')
ax8.set_title('crime')
datacorr['gdp'] = np.log(datacorr['gdp'])
datacorr['crime'] = np.log(datacorr['crime'])
datacorr['infant_mortality'] = np.log(datacorr['infant_mortality']) 
datacorr['corruption'] = datacorr['corruption']**0.5
datacorr['pressfreedom'] = datacorr['pressfreedom']**0.5
fig = plt.figure(figsize=(10,8))
ax1= fig.add_subplot(241)
ax1.hist(datacorr.happy, color='blue')
ax1.set_title('happiness')
ax2= fig.add_subplot(242)
ax2.hist(datacorr.gdp, color='blue')
ax2.set_title('gdp')
ax3= fig.add_subplot(243)
ax3.hist(datacorr.life_expectancy, color='blue')
ax3.set_title('life_expectancy')
ax4= fig.add_subplot(244)
ax4.hist(datacorr.infant_mortality, color='blue')
ax4.set_title('infant_mortality')
ax5= fig.add_subplot(245)
ax5.hist(datacorr.pressfreedom, color='blue')
ax5.set_title('pressfreedom')
ax6= fig.add_subplot(246)
ax6.hist(datacorr.corruption, color='blue')
ax6.set_title('corruption')
ax7= fig.add_subplot(247)
ax7.hist(datacorr.inequality, color='blue')
ax7.set_title('inequality')
ax8= fig.add_subplot(248)
ax8.hist(datacorr.crime, color='blue')
ax8.set_title('crime')

#Data transformations:
dataset['gdp2014'] = np.log(dataset['gdp2014'])
dataset['gdp2015'] = np.log(dataset['gdp2015'])
dataset['gdp2016'] = np.log(dataset['gdp2016'])
dataset['gdp_chg']=100*(dataset['gdp2016']/dataset['gdp2015']-1)
dataset['crime'] = np.log(dataset['murder2016']) 
dataset['life_expectancy_chg']=100*(dataset['life2015']/dataset['life2014']-1)
dataset['press2016'] = dataset['press2016']**0.5
dataset['pressfreedom_chg']=100*(dataset['press2016']/dataset['press2015']-1)
dataset['corrupt2016'] = dataset['corrupt2016']**0.5
dataset['corrupt2015'] = dataset['corrupt2015']**0.5
dataset['corruption_chg']=100*(dataset['corrupt2016']/dataset['corrupt2015']-1)
dataset['infant2016'] = np.log(dataset['infant2016'])
dataset['infant2015'] = np.log(dataset['infant2015'])
dataset['infant_mortality_chg']=100*(dataset['infant2016']/dataset['infant2015']-1)
#dataset.to_csv('data_corr_chg.csv')
datacorrchg = pd.read_csv('data_corr_chg.csv')

#Correlation matrix
infert = com.importr("ISLR")
corr = datacorr.corr()
sns.heatmap(corr)

#Correlation matrix
corr = datacorrchg.corr()
sns.heatmap(corr)

#Matrix of scatterplots:
fig = plt.figure(figsize=(16,10))
ax1= fig.add_subplot(241)
ax1.scatter(dataset['gdp2016'], dataset['happy2017'], color='blue')
ax1.set_title('happiness vs GDP, corr=0.82')
ax2= fig.add_subplot(242)
ax2.scatter(dataset['life2015'], dataset['happy2017'], color='blue')
ax2.set_title('happiness vs life expectancy, corr=0.79')
ax3= fig.add_subplot(243)
ax3.scatter(dataset['infant2016'], dataset['happy2017'], color='blue')
ax3.set_title('happiness vs infant mortality, corr=-0.76')
ax4= fig.add_subplot(244)
ax4.scatter(dataset['press2016'], dataset['happy2017'], color='blue')
ax4.set_title('happiness vs press freedom, corr=-0.39')
ax5= fig.add_subplot(245)
ax5.scatter(dataset['corrupt2016'], dataset['happy2017'], color='blue')
ax5.set_title('happiness vs corruption, corr=0.71')
ax6= fig.add_subplot(246)
ax6.scatter(dataset['crime'], dataset['happy2017'], color='blue')
ax6.set_title('happiness vs murder rate, corr=-0.32')
ax7= fig.add_subplot(247)
ax7.scatter(dataset['life_expectancy_chg'], dataset['happy2017'], color='blue')
ax7.set_title('happiness vs life expectancy change, corr=-0.56')
ax8= fig.add_subplot(248)
ax8.scatter(dataset['pressfreedom_chg'], dataset['happy2017'], color='blue')
ax8.set_title('happiness vs press freedom change, corr=0.49')

regressors = dataset.as_matrix(['gdp', 'life_expectancy', 'infant_mort', 'pressfreedom', 'corruption', 'crime'])
corr = np.corrcoef(regressors, rowvar=0)  # correlation matrix
w, v = np.linalg.eig(corr)        # eigen values & eigen vectors
w


dataset = dataset.rename(columns={'happy2017':'happy', 'gdp2016': 'gdp', 'life2015': 'life_expectancy', 'infant2016':'infant_mort', 'press2016':'pressfreedom', 'corrupt2016': 'corruption'})
dataset.to_csv('data_for_regression.csv')

