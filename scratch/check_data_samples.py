import pandas as pd
hh = pd.read_excel("noida_electricity_household.xlsx")
comm = pd.read_excel("noida_commercial.xlsx")
print("Household Sector Sample:")
print(hh['Sector'].head(10).tolist())
print("\nCommercial Area Sample:")
print(comm['Area'].head(10).tolist())
