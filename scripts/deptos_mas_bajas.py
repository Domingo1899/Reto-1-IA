"""Investiga Salto y Paysandú: ¿de dónde viene su caída extrema de intensidad?"""
import unicodedata
from pathlib import Path
import pandas as pd
 
CARPETA = Path("data/interim/entrega-02-corregida")
MESES = ["dias4", "dias5", "dias6"]
FOCO = ["salto", "paysandu"]
 
def norm(v):
    if pd.isna(v): return ""
    s = str(v).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))
 
def preparar(df, pob, anio):
    df = df.copy()
    df.columns = [norm(c).replace(" ", "_") for c in df.columns]
    for c in ["zona","rubro","dept_nombre","tipo_centro"]:
        if c in df.columns: df[c] = df[c].map(norm)
    df = df.drop_duplicates()
    for c in MESES: df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["dias_total"] = df[MESES].sum(axis=1)
    df["anio"], df["poblacion"] = anio, pob
    return df
 
def caida(df, dims, umbral=10):
    """Tasa de acceso intenso (>=umbral días) por dimensión, 2025 vs 2026.
 
    Devuelve, para cada fila, la cantidad de gente en NÚMERO absoluto
    (n_2025/n_2026 = base total, i_2025/i_2026 = cuántos accedieron
    intenso) además del % (tasa_2025/tasa_2026) y su variación.
    """
    df = df.copy()
    df["intenso"] = df["dias_total"] >= umbral
    g = df.groupby(dims+["anio"]).agg(n=("intenso","size"), i=("intenso","sum")).reset_index()
    g["tasa"] = (100*g["i"]/g["n"]).round(1)
 
    piv = g.pivot_table(index=dims, columns="anio", values=["n", "i", "tasa"]).reset_index()
    piv.columns = [f"{a}_{b}" if b != "" else a for a, b in piv.columns]
    piv.columns.name = None
 
    for c in ("n_2025", "n_2026", "i_2025", "i_2026"):
        if c in piv:
            piv[c] = piv[c].fillna(0).astype(int)
 
    if "tasa_2025" in piv and "tasa_2026" in piv:
        piv["var_pp"] = (piv["tasa_2026"] - piv["tasa_2025"]).round(1)
 
    cols = dims + [c for c in ("n_2025", "n_2026", "i_2025", "i_2026",
                                "tasa_2025", "tasa_2026", "var_pp") if c in piv]
    piv = piv[cols]
    return piv.sort_values("var_pp") if "var_pp" in piv else piv
 
def main():
    partes=[]
    for r in sorted(CARPETA.glob("*.parquet")):
        pob,anio=r.stem.split("_"); partes.append(preparar(pd.read_parquet(r),pob,int(anio)))
    df=pd.concat(partes,ignore_index=True)
    foco=df[df.dept_nombre.isin(FOCO)]
 
    for pob in ["estudiantes","docentes"]:
        sub=foco[foco.poblacion==pob]
        print(f"\n{'#'*60}\n{pob.upper()} en SALTO y PAYSANDÚ\n{'#'*60}")
        print("\n-- por depto x subsistema --")
        print(caida(sub,["dept_nombre","rubro"]).to_string(index=False))
        print("\n-- por depto x zona --")
        print(caida(sub,["dept_nombre","zona"]).to_string(index=False))
        print("\n-- por depto x tipo de centro --")
        print(caida(sub,["dept_nombre","tipo_centro"]).to_string(index=False))
 
if __name__=="__main__": main()
 