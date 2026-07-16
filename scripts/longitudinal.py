"""c) Los MISMOS usuarios en 2025 y 2026: ¿cómo cambió su frecuencia individual?
Solo posible ahora que los IDs son estables (ver chequeo_ids con datos corregidos)."""
import unicodedata
from pathlib import Path
import pandas as pd

CARPETA = Path("data/interim/entrega-02-corregida")
MESES = ["dias4", "dias5", "dias6"]

def norm(v):
    if pd.isna(v): return ""
    return str(v).strip().lower()

def preparar(df):
    df = df.copy()
    df.columns = [norm(c).replace(" ", "_") for c in df.columns]
    df = df.drop_duplicates()
    for c in MESES: df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["dias_total"] = df[MESES].sum(axis=1)
    # una fila por persona (sumamos días si tiene varias filas, ej docentes)
    return df.groupby("id_persona", as_index=False)["dias_total"].sum()

def main():
    for pob in ["docentes", "estudiantes"]:
        a = preparar(pd.read_parquet(CARPETA/f"{pob}_2025.parquet"))
        b = preparar(pd.read_parquet(CARPETA/f"{pob}_2026.parquet"))

        # unir por ID: solo los que están en AMBOS años
        m = a.merge(b, on="id_persona", suffixes=("_25","_26"))
        print(f"\n{'='*60}\nLONGITUDINAL — {pob.upper()}\n{'='*60}")
        print(f"Usuarios presentes en ambos años: {len(m):,}")

        # clasificamos cada usuario según qué le pasó
        def clasificar(r):
            u25 = r.dias_total_25 >= 1
            u26 = r.dias_total_26 >= 1
            if u25 and not u26: return "dejó de entrar"
            if not u25 and u26: return "empezó a entrar"
            if u25 and u26:
                if r.dias_total_26 < r.dias_total_25 * 0.7: return "bajó intensidad"
                if r.dias_total_26 > r.dias_total_25 * 1.3: return "subió intensidad"
                return "se mantuvo"
            return "nunca entró"
        m["cambio"] = m.apply(clasificar, axis=1)

        res = m["cambio"].value_counts()
        pct = (100*res/len(m)).round(1)
        for k in res.index:
            print(f"  {k:<18} {res[k]:>8,}  ({pct[k]}%)")

        print(f"\n  Días promedio 2025: {m.dias_total_25.mean():.1f} | "
              f"2026: {m.dias_total_26.mean():.1f}")

if __name__ == "__main__":
    main()