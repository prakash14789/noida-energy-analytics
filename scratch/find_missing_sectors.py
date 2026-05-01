import pandas as pd
import os

# Copy the loc_map from ui_tabs.py (partial)
loc_map_keys = [
    'Gamma 1', 'Gamma 2', 'Delta 1', 'Delta 2', 'Alpha 1', 'Alpha 2', 'Beta 1', 'Beta 2',
    'Zeta 1', 'Zeta 2', 'Eta 1', 'Eta 2', 'Phi 1', 'Phi 2', 'Omega 1', 'Omega 2',
    'Sigma 1', 'Sigma 2', 'Pi 1', 'Pi 2', 'Knowledge Park I', 'Knowledge Park II',
    'Knowledge Park III', 'Sector 150', 'Sector 1', 'Sector 2', 'Sector 18', 'Sector 62',
    'Sector 63', 'Sector 137', 'Sector 15', 'Sector 16', 'Sector 50', 'Sector 75',
    'Sector 132'
]

hh = pd.read_excel("noida_electricity_household.xlsx")
comm = pd.read_excel("noida_commercial.xlsx")

hh_sectors = hh['Sector'].unique()
comm_areas = comm['Area'].unique()

all_sectors = set(hh_sectors) | set(comm_areas)

missing = [s for s in all_sectors if s not in loc_map_keys]
print(f"Total Unique Sectors/Areas: {len(all_sectors)}")
print(f"Missing from loc_map: {missing}")
