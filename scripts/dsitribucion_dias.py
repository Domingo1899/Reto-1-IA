"""b) Distribución de días de acceso: ¿la curva se corre a la izquierda en 2026?"""
import unicodedata
from pathlib import Path
import pandas as pd

CARPETA = Path("data/interim/entrega-02-corregida")
MESES = ["dias4", "dias5", "dias6"]

def norm(v):
    if pd.isna(v): return ""
    return str(v).strip().lower()

def preparar(df, poblacion, anio):
    df = df.copy()
    df.columns = [norm(c).replace(" ", "_") for c in df.columns]
    df = df.drop_duplicates()
    for c in MESES: df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["dias_total"] = df[MESES].sum(axis=1)
    df["anio"], df["poblacion"] = anio, poblacion
    return df

def main():
    partes = []
    for r in sorted(CARPETA.glob("*.parquet")):
        pob, anio = r.stem.split("_")
        partes.append(preparar(pd.read_parquet(r), pob, int(anio)))
    df = pd.concat(partes, ignore_index=True)

    # agrupamos en rangos para que se lea fácil
    bins = [-1, 0, 5, 15, 30, 60, 999]
    labels = ["0 dias", "1-5", "6-15", "16-30", "31-60", "61+"]
    df["rango"] = pd.cut(df["dias_total"], bins=bins, labels=labels)

    for pob in ["docentes", "estudiantes"]:
        sub = df[df.poblacion == pob]
        print(f"\n{'='*60}\nDISTRIBUCIÓN DE DÍAS — {pob.upper()}\n{'='*60}")
        t = pd.crosstab(sub["rango"], sub["anio"], normalize="columns")*100
        t = t.round(1)
        t["var_pp"] = (t[2026]-t[2025]).round(1)
        print(t.to_string())
        print("  (var_pp positivo en '0 dias'/'1-5' + negativo en rangos altos = desenganche)")

if __name__ == "__main__":
    main()