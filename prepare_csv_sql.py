import pandas as pd
from pathlib import Path
import unicodedata

BASE_DIR = Path(__file__).parent

def normalize_col(col: str) -> str:
    col = str(col).strip()
    col = unicodedata.normalize("NFKD", col).encode("ascii", "ignore").decode("utf-8")
    col = col.replace(" ", "_")
    return col

# ------- data.csv -------
df_data = pd.read_csv(BASE_DIR / "data.csv")
df_data.columns = [normalize_col(c) for c in df_data.columns]

df_data = df_data.rename(columns={
    "code": "code_position",
    "PORTEFEUILLE": "code_fonds",
    "TITRE": "code_titre",
    "DESCRIPTION": "description_titre",
    "CLASSE": "classe_titre",
    "Quantite_Actif": "quantite_actif",
    "COURS": "cours",
    "Valo_Titre_CV": "valo_titre_cv",
    "POIDS": "poids_pct",
    "ACTIF_NET_NET": "actif_net"
})

df_data["code_fonds"] = df_data["code_fonds"].astype(str).str.strip()
df_data["code_titre"] = df_data["code_titre"].astype(str).str.strip()

for col in ["quantite_actif", "cours", "valo_titre_cv", "actif_net"]:
    df_data[col] = pd.to_numeric(df_data[col], errors="coerce")

df_data["poids"] = (
    df_data["poids_pct"]
    .astype(str)
    .str.replace("%", "", regex=False)
    .str.replace(",", ".", regex=False)
    .astype(float) / 100.0
)

df_data.to_csv(BASE_DIR / "data_clean.csv", index=False)

# ------- z classification.csv -------
df_classif = pd.read_csv(BASE_DIR / "z classification.csv")
df_classif.columns = [normalize_col(c) for c in df_classif.columns]

df_classif = df_classif.rename(columns={
    "CODE_": "code_isin",
    "OPCVM": "nom_opcvm",
    "Nature_WG": "nature_wg",
    "Nature_juridique": "nature_juridique",
    "Classification": "classification",
    "Depositaire": "depositaire",
    "AN": "actif_net",
    "Nature_WG.1": "nature_wg_groupe",
    "Code": "code_fonds"
})

if "nature_wg_groupe" not in df_classif.columns and "Nature_WG.1" in df_classif.columns:
    df_classif = df_classif.rename(columns={"Nature_WG.1": "nature_wg_groupe"})
if "nature_wg_groupe" not in df_classif.columns:
    df_classif["nature_wg_groupe"] = None

mask_header = df_classif["code_isin"].astype(str).str.startswith("CODE")
df_classif = df_classif[~mask_header]
df_classif = df_classif[df_classif["code_fonds"].notna()]

df_classif["code_fonds"] = df_classif["code_fonds"].astype(str).str.strip()
df_classif["actif_net"] = (
    df_classif["actif_net"]
    .astype(str)
    .str.replace(",", ".", regex=False)
)
df_classif["actif_net"] = pd.to_numeric(df_classif["actif_net"], errors="coerce")

df_classif.to_csv(BASE_DIR / "z_classification_clean.csv", index=False)

# ------- z poids actions.csv -------
df_poids = pd.read_csv(BASE_DIR / "z poids actions.csv")
df_poids.columns = [normalize_col(c) for c in df_poids.columns]

df_poids = df_poids.rename(columns={
    "Code": "code_fonds",
    "Fonds_": "nom_fonds",
    "Poids_Actions": "poids_actions_pct"
})

df_poids = df_poids[df_poids["code_fonds"].notna()]
df_poids["code_fonds"] = df_poids["code_fonds"].astype(str).str.strip()

if "poids_actions_pct" in df_poids.columns:
    df_poids["poids_actions"] = (
        df_poids["poids_actions_pct"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", ".", regex=False)
        .astype(float) / 100.0
    )

df_poids.to_csv(BASE_DIR / "z_poids_actions_clean.csv", index=False)

print("CSV nettoyés générés : data_clean.csv, z_classification_clean.csv, z_poids_actions_clean.csv")