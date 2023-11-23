import pandas as pd

path = 'Year and population.xlsx'

datas = pd.read_excel(path)

datas.to_csv('Year and population.csv',index=False)
