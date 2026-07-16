"""a) ¿Dónde se concentra la caída de intensidad? Umbral >=10 días por segmento."""
import unicodedata
from pathlib import Path
import pandas as pd

CARPETA = Path("data/interim/entrega-02-corregida")
MESES = ["dias4", "dias5", "dias6"]

def norm(v):
    if pd.isna(v): return ""
    s = str(v).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))

def preparar(df, poblacion, anio):
    df = df.copy()
    df.columns = [norm(c).replace(" ", "_") for c in df.columns]
    for col in ["zona","tipo_centro","rubro","dept_nombre","sexo"]:
        if col in df.columns: df[col] = df[col].map(norm)
    df = df.drop_duplicates()
    for c in MESES: df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["dias_total"] = df[MESES].sum(axis=1)
    df["anio"], df["poblacion"] = anio, poblacion
    return df

def caida_intensidad(df, dim, umbral=10):
    """% con >=umbral días, por segmento y año, con la variación."""
    df = df.copy()
    df["intenso"] = df["dias_total"] >= umbral
    g = df.groupby(["poblacion", dim, "anio"]).agg(
        n=("intenso","size"), intensos=("intenso","sum")).reset_index()
    g["tasa"] = (100*g["intensos"]/g["n"]).round(1)
    piv = g.pivot_table(index=["poblacion",dim], columns="anio", values="tasa").reset_index()
    piv.columns.name = None
    piv["var_pp"] = (piv[2026]-piv[2025]).round(1)
    return piv.sort_values("var_pp")

def main():
    partes = []
    for r in sorted(CARPETA.glob("*.parquet")):
        pob, anio = r.stem.split("_")
        partes.append(preparar(pd.read_parquet(r), pob, int(anio)))
    df = pd.concat(partes, ignore_index=True)

    for dim in ["rubro", "dept_nombre", "zona", "sexo"]:
        print(f"\n{'='*70}\nCAÍDA DE INTENSIDAD (>=10 días) POR {dim.upper()}\n{'='*70}")
        print(caida_intensidad(df, dim).to_string(index=False))

if __name__ == "__main__":
    main()