import pandas as pd

dataFrame = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
print(dataFrame, "\n")
dataFrame = pd.DataFrame(
    {'a': [1, 2, 3], 'b': [4, 5, 6]}, index=['x', 'y', 'z'])
print(dataFrame, "\n")

dataFrame = pd.DataFrame(
    {'Nombre': ["Juan", "María", "Pedro", "José"], 'Edad': [20, 26, 18, 22], 'País': ["Argentina", "Peru", "Brasil", "Chile"]})
print(dataFrame, "\n")
for col in dataFrame.columns:
    print(col)
# Add a new row
new_row = {'Nombre': 'Pablo', 'Edad': 30, 'País': 'Colombia'}
new_row_df = pd.DataFrame([new_row])
dataFrame = pd.concat([dataFrame, new_row_df], ignore_index=True)
# or
dataFrame.loc[len(dataFrame)] = new_row
print(dataFrame)

# remove repeated 'Pablo'
dataFrame = dataFrame.drop_duplicates(subset=['Nombre'], keep='last')
print(dataFrame)

# rename attributes with value 'Peru' to 'Perú'
dataFrame = dataFrame.replace(to_replace='Peru', value='Perú')
print(dataFrame)
