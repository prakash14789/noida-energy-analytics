import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np
import os

def fetch_data():
    # Place name
    place = "Greater Noida"
    print(f"Attempting to fetch data for {place}...")
    
    # Try to get the bounding box of the place
    try:
        # 🏢 Buildings using a bounding box around a center point to be safe
        # Center: 28.474, 77.507
        dist = 3000 # 3km radius
        center_point = (28.474, 77.507)
        
        print(f"Fetching buildings within {dist}m of {center_point}...")
        try:
            buildings = ox.features_from_point(center_point, tags={"building": True}, dist=dist)
        except AttributeError:
            buildings = ox.geometries_from_point(center_point, tags={"building": True}, dist=dist)
            
        # Keep only polygons and multipolygons
        buildings = buildings[buildings.geometry.type.isin(["Polygon", "MultiPolygon"])]
        buildings = buildings.reset_index()
        
        # Keep useful columns
        cols = ["geometry", "building", "name"]
        existing_cols = [c for c in cols if c in buildings.columns]
        buildings = buildings[existing_cols]
        buildings = buildings.dropna(subset=["geometry"])
        
        # Assign height (simulated as requested)
        def assign_height(x):
            if x == "commercial": return np.random.randint(40, 100)
            elif x == "residential": return np.random.randint(20, 60)
            else: return np.random.randint(10, 40)
        
        buildings["height"] = buildings["building"].apply(assign_height)
        
        # Ensure CRS is EPSG:4326 for Pydeck
        buildings = buildings.to_crs(epsg=4326)
        
        # Save to GeoJSON
        buildings.to_file("buildings_gn.geojson", driver='GeoJSON')
        print(f"Buildings saved to buildings_gn.geojson ({len(buildings)} buildings)")
    except Exception as e:
        print(f"Error fetching buildings: {e}")

    # 🛣 Roads
    try:
        print(f"Fetching road network for {place}...")
        # Try place first, then point if it fails
        try:
            graph = ox.graph_from_place(place, network_type="drive")
        except Exception:
            graph = ox.graph_from_point(center_point, dist=dist, network_type="drive")
            
        roads = ox.graph_to_gdfs(graph, nodes=False)
        roads = roads.reset_index()
        
        # Ensure CRS is EPSG:4326
        roads = roads.to_crs(epsg=4326)
        
        # Keep only useful columns
        cols = ["geometry", "name", "highway"]
        existing_cols = [c for c in cols if c in roads.columns]
        roads = roads[existing_cols]
        
        # Save to GeoJSON
        roads.to_file("roads_gn.geojson", driver='GeoJSON')
        print(f"Roads saved to roads_gn.geojson ({len(roads)} road segments)")
    except Exception as e:
        print(f"Error fetching roads: {e}")

if __name__ == "__main__":
    fetch_data()
