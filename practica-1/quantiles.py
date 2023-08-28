import pandas as pd
import os
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'Drug5_atipicos.csv')

df = pd.read_csv(file_path)

print(df.describe())

dfNormal = df.Age[df.Cholesterol == 'NORMAL']
QNormal = dfNormal.quantile([0.25, 0.5, 0.75]).values
print("CUARTILES - Edades c/ Colesterol Normal")
print(QNormal)

dfHigh = df.Age[df.Cholesterol == 'HIGH']
QHigh = dfHigh.quantile([0.25, 0.5, 0.75]).values
print("CUARTILES - Edades c/ Colesterol Alto")
print(QHigh)
