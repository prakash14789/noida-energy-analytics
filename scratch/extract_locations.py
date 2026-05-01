import pandas as pd
hh = pd.read_excel("noida_electricity_household.xlsx")
comm = pd.read_excel("noida_commercial.xlsx")
print("Household Sectors:")
print(hh['Sector'].unique())
print("\nCommercial Areas:")
print(comm['Area'].unique())
