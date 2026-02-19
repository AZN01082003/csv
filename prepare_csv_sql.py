import pandas as pd
from pathlib import Path
import unicodedata
import re

BASE_DIR = Path(__file__).parent


def normalize_col(col: str) -> str:
    col = str(col).strip()
    col = unicodedata.normalize("NFKD", col).encode("ascii", "ignore").decode("utf-8")
    col = re.sub(r"[^a-zA-Z0-9]+", "_", col)
    col = col.strip("_").lower()
    return col


# ─────────────────────────────────────────────
# 1. data.csv
# ─────────────────────────────────────────────
# Read everything as str to preserve leading zeros (e.g. "001")
df_data = pd.read_csv(BASE_DIR / "data.csv", dtype=str)
df_data.columns = [normalize_col(c) for c in df_data.columns]

# After normalize_col the header becomes:
#   code, portefeuille, titre, description, classe, quantite_actif, cours,
#   valo_titre_cv, poids, actif_net_net
df_data = df_data.rename(columns={
    "code":           "code_position",
    "portefeuille":   "code_fonds",
    "titre":          "code_titre",
    "description":    "description_titre",
    "classe":         "classe_titre",
    "quantite_actif": "quantite_actif",
    "cours":          "cours",
    "valo_titre_cv":  "valo_titre_cv",
    "poids":          "poids_pct",
    "actif_net_net":  "actif_net",
})

# Preserve leading zeros: pad code_fonds to 3 chars
df_data["code_fonds"] = (
    df_data["code_fonds"].astype(str).str.strip().str.zfill(3)
)
df_data["code_titre"]        = df_data["code_titre"].astype(str).str.strip()
df_data["description_titre"] = df_data["description_titre"].astype(str).str.strip()
df_data["classe_titre"]      = df_data["classe_titre"].astype(str).str.strip()

# Convert numeric columns
for col in ["cours", "valo_titre_cv", "actif_net"]:
    df_data[col] = pd.to_numeric(
        df_data[col].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )

# quantite_actif est toujours un entier → Int64 nullable (écrit sans ".0" dans le CSV)
df_data["quantite_actif"] = (
    pd.to_numeric(
        df_data["quantite_actif"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )
    .round(0)
    .astype("Int64")
)

# Convert poids: "1.2741830740926%" → 0.012741830740926
# Attention : certaines valeurs sont très petites (ex: 0.000000474940627922239%)
# → Python les écrit en notation scientifique (4.749e-09) que SSMS ne sait pas lire.
# On les formate en virgule fixe avec 10 décimales pour éviter ce problème.
df_data["poids"] = (
    df_data["poids_pct"]
    .astype(str)
    .str.replace("%", "", regex=False)
    .str.replace(",", ".", regex=False)
    .str.strip()
    .pipe(pd.to_numeric, errors="coerce")
    .div(100.0)
    # Formater en virgule fixe (10 décimales) → évite "4.749e-09"
    .apply(lambda x: f"{x:.10f}" if pd.notna(x) else "")
)

# Drop the raw percentage column — keep only the decimal value
df_data = df_data.drop(columns=["poids_pct"])

# float_format='%.6f' pour cours / valo / actif_net → pas de notation scientifique
df_data.to_csv(
    BASE_DIR / "data_clean.csv",
    index=False,
    encoding="utf-8-sig",
    float_format="%.6f",
)
print(f"data_clean.csv      → {len(df_data)} lignes")


# ─────────────────────────────────────────────
# 2. z classification.csv
# ─────────────────────────────────────────────
# The file has repeated header rows acting as section separators.
# "Nature WG" appears twice → pandas deduplicates to "Nature WG" / "Nature WG.1"
# "CODE " (col 0) and "Code" (col 8) would both normalize to "code" causing
# duplicate column names — so we assign explicit SQL-friendly names directly.
df_classif = pd.read_csv(BASE_DIR / "z classification.csv", dtype=str)
df_classif.columns = [
    "code_isin", "nom_opcvm", "nature_wg", "nature_juridique",
    "classification", "depositaire", "actif_net", "nature_wg_groupe", "code_fonds",
]

# Remove repeated header rows (code_isin column contains "CODE " i.e. the literal header)
mask_header = df_classif["code_isin"].astype(str).str.upper().str.strip().isin(["CODE", "CODE_"])
df_classif = df_classif[~mask_header]

# Remove rows without a fund code
df_classif = df_classif[df_classif["code_fonds"].notna()]
df_classif = df_classif[df_classif["code_fonds"].astype(str).str.strip() != ""]

# Pad code_fonds to 3 chars
df_classif["code_fonds"] = (
    df_classif["code_fonds"].astype(str).str.strip().str.zfill(3)
)

# Trim string columns
for col in ["code_isin", "nom_opcvm", "nature_wg", "nature_juridique",
            "classification", "depositaire", "nature_wg_groupe"]:
    if col in df_classif.columns:
        df_classif[col] = df_classif[col].astype(str).str.strip()

# Convert actif_net to numeric
df_classif["actif_net"] = pd.to_numeric(
    df_classif["actif_net"].astype(str).str.replace(",", ".", regex=False),
    errors="coerce",
)

df_classif.to_csv(
    BASE_DIR / "z_classification_clean.csv", index=False, encoding="utf-8-sig"
)
print(f"z_classification_clean.csv → {len(df_classif)} lignes")


# ─────────────────────────────────────────────
# 3. z poids actions.csv
# ─────────────────────────────────────────────
# Header: "Code,Fonds ," — trailing comma creates an unnamed 3rd column.
# Section-header rows have Code=NaN and Fonds=<group name> (e.g. "ACTIONS GP").
# Data rows: Code=3-digit string, Fonds=fund name, col3=percentage or empty.
df_poids = pd.read_csv(BASE_DIR / "z poids actions.csv", dtype=str, header=0)
df_poids.columns = [normalize_col(c) for c in df_poids.columns]

# After normalize_col: "Code"→"code", "Fonds "→"fonds", unnamed col→varies
# Identify the actual column names
col_code  = df_poids.columns[0]   # "code"
col_fonds = df_poids.columns[1]   # "fonds"
col_poids = df_poids.columns[2]   # unnamed / "poids_actions" / etc.

# Propagate group name from section-header rows to subsequent data rows
groups = []
current_group = None
for _, row in df_poids.iterrows():
    code_val = str(row[col_code]).strip()
    if code_val in ("", "nan"):
        # This is a section-header row — extract group name
        current_group = str(row[col_fonds]).strip()
        groups.append(None)          # will be removed later
    else:
        groups.append(current_group)

df_poids["groupe"] = groups

# Remove section-header rows (code is empty / NaN)
df_poids = df_poids[
    df_poids[col_code].notna() &
    (df_poids[col_code].astype(str).str.strip() != "") &
    (df_poids[col_code].astype(str).str.strip() != "nan")
]

# Rename columns
df_poids = df_poids.rename(columns={
    col_code:  "code_fonds",
    col_fonds: "nom_fonds",
    col_poids: "poids_actions_pct",
})

# Pad code_fonds to 3 chars
df_poids["code_fonds"] = (
    df_poids["code_fonds"].astype(str).str.strip().str.zfill(3)
)
df_poids["nom_fonds"] = df_poids["nom_fonds"].astype(str).str.strip()

# Convert poids: "100%" → 1.0, "30%" → 0.3, "" → NaN
df_poids["poids_actions"] = (
    df_poids["poids_actions_pct"]
    .astype(str)
    .str.replace("%", "", regex=False)
    .str.replace(",", ".", regex=False)
    .str.strip()
    .replace("nan", "")
    .pipe(pd.to_numeric, errors="coerce")
    .div(100.0)
)

# Drop raw percentage column
df_poids = df_poids.drop(columns=["poids_actions_pct"])

df_poids.to_csv(
    BASE_DIR / "z_poids_actions_clean.csv", index=False, encoding="utf-8-sig"
)
print(f"z_poids_actions_clean.csv  → {len(df_poids)} lignes")

print("\nCSV nettoyés générés avec succès.")
