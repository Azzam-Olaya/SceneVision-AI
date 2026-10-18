import pandas as pd
import seaborn as sns
import pathlib as Path
import matplotlib.pyplot as plt

Dossier_Name = Path("data/raw")
PATH = Dossier_Name / "seg_train"
DOSSIER_CLS = {
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street"
}




