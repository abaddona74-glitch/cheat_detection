import pickle
import pandas as pd
from pandasgui import show

with open("violations_database.pkl", "rb") as f:
    data = pickle.load(f)

# Agar data pandas DataFrame bo‘lsa:
show(data)
