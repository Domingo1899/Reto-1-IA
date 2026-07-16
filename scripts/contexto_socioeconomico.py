"""¿Los centros más vulnerables (quintil 1) cayeron más que los favorecidos (5)?
CONTEXTO = primaria (trae zona: Urbano/Rural). IVSMEDIA = media, pero SOLO 2026."""
import unicodedata, re
from pathlib import Path
import pandas as pd

CARPETA = Path("data/interim/entrega-02-corregida")
MESES = ["dias4", "dias5", "dias6"]

def norm(v):
    if pd.isna(v): return ""
    s=str(v).strip().lower()
    s=unicodedata.normalize("NFKD",s)
    return "".join(c for c in s if not unicodedata.combining(c))

def preparar(df,pob,anio):
    df=df.copy()
    df.columns=[norm(c).replace(" ","_") for c in df.columns]
    df=df.rename(columns={"contexto_sociocultural":"contexto"})
    for c in ["rubro","contexto","ivsmedia"]:
        if c in df.columns: df[c]=df[c].map(norm)
    df=df.drop_duplicates()
    for c in MESES: df[c]=pd.to_numeric(df[c],errors="coerce").fillna(0)
    df["dias_total"]=df[MESES].sum(axis=1)
    df["anio"],df["poblacion"]=anio,pob
    return df

def quintil(v):
    """extrae el número de quintil de 'quintil urbano 3' o 'quintil 3'."""
    m=re.search(r"(\d)", str(v))
    return int(m.group()) if m else None

def zona_ctx(v):
    if "urbano" in str(v): return "urbano"
    if "rural" in str(v): return "rural"
    return "otro"

def caida_por_quintil(df, col, umbral=10):
    df=df.copy()
    df=df[df[col].notna() & (df[col].map(norm)!="")]
    df["q"]=df[col].map(quintil)
    df=df[df["q"].notna()]
    df["intenso"]=df["dias_total"]>=umbral
    g=df.groupby(["q","anio"]).agg(n=("intenso","size"),i=("intenso","sum")).reset_index()
    g["tasa"]=(100*g["i"]/g["n"]).round(1)
    piv=g.pivot_table(index="q",columns="anio",values="tasa").reset_index()
    piv.columns.name=None
    if 2025 in piv and 2026 in piv: piv["var_pp"]=(piv[2026]-piv[2025]).round(1)
    return piv

def main():
    partes=[]
    for r in sorted(CARPETA.glob("*.parquet")):
        pob,anio=r.stem.split("_"); partes.append(preparar(pd.read_parquet(r),pob,int(anio)))
    df=pd.concat(partes,ignore_index=True)

    print("="*60)
    print("CONTEXTO (PRIMARIA) — caída de intensidad por quintil")
    print("quintil 1 = MÁS vulnerable | 5 = menos")
    print("="*60)
    est=df[df.poblacion=="estudiantes"]
    # separamos urbano y rural (son escalas distintas)
    est=est.copy(); est["zona_ctx"]=est["contexto"].map(zona_ctx)
    for z in ["urbano","rural"]:
        print(f"\n-- primaria {z} --")
        print(caida_por_quintil(est[est.zona_ctx==z],"contexto").to_string(index=False))

    print("\n"+"="*60)
    print("IVSMEDIA (SECUNDARIA/UTU) — SOLO existe en 2026")
    print("No hay comparación 2025 vs 2026. Solo foto transversal 2026.")
    print("="*60)
    m26=est[est.anio==2026].copy()
    m26=m26[m26.ivsmedia.map(norm).str.contains("quintil",na=False)]
    m26["q"]=m26.ivsmedia.map(quintil)
    m26["intenso"]=m26.dias_total>=10
    g=m26.groupby("q").agg(n=("intenso","size"),tasa=("intenso","mean")).reset_index()
    g["tasa"]=(100*g["tasa"]).round(1)
    print(g.to_string(index=False))

if __name__=="__main__": main()