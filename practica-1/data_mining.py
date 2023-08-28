import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'Premios2020.csv')

df = pd.read_csv(file_path, encoding='ISO-8859-1')
print(df)
print(df.describe(), '\n')
print(df.isnull().sum(), '\n')
print(pd.value_counts(df['genre1']))

print('---------------------------------------------------')
df2 = pd.DataFrame(df)
modeGenre = df2.genre2.mode()
df2['genre2'] = df2['genre2'].fillna(modeGenre[0])
print(df2.isnull().sum())

print('---------------------------------------------------')
df3 = pd.DataFrame(df)
df3['genre1'] = df3['genre1'].replace(
    ['Adventure', 'Action', 'Romance', 'Thriller', 'Mystery'], 'Otra')
print(pd.value_counts(df3['genre1']))

print('---------------------------------------------------')
df4 = pd.DataFrame(df)
etiq = ["low", "mid", "high", "very high"]
# Discretización por RANGO
column = pd.cut(df4["duration"], bins=len(etiq), labels=etiq)
df4['durationRange'] = pd.Series.to_frame(column)
# Discretización por FRECUENCIA
column2 = pd.qcut(df4["duration"], q=len(etiq), labels=etiq)
df4['durationFreq'] = pd.Series.to_frame(column2)
print(df4[['duration', 'durationRange', 'durationFreq']], '\n')
print(pd.value_counts(df4['durationRange']), '\n')
print(pd.value_counts(df4['durationFreq']))
