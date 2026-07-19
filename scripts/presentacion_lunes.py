"""
presentacion.py — Las 4 láminas para Ceibal (20/07/2026).

Genera dashboards/presentacion.html: un reporte corto, pensado para proyectar,
con UNA historia y UN giro.

  1. No es que entren menos, es que entran menos veces  (el titular)
  2. Entre las mismas personas, también cae             (blinda contra "es rotación")
  3. Se soltó el núcleo obligatorio                      (dónde ocurre)
  4. Dónde poner los recursos                            (volumen, no porcentaje)

POR QUÉ ESTAS CUATRO Y NO OTRAS
El resto de los grupos midió tasa de acceso y días promedio. Nadie midió cómo
cambia la caída al exigir MÁS intensidad. Ese barrido de umbrales es el aporte
diferencial, y las otras tres láminas existen para sostenerlo: la 2 responde la
objeción de rotación de matrícula, la 3 dice dónde pasa y la 4 dice qué hacer.

Se dejan afuera a propósito territorio, meses, tipo de centro y quintil
socioeconómico: los cubrieron otros grupos y el quintil tiene una discusión
abierta (efecto piso + composición + maduración) que no conviene abrir en vivo.

Lee todo de los parquet, sin depender de los CSV intermedios.
"""

import unicodedata
from pathlib import Path

import pandas as pd
import plotly.express as px

CARPETA = Path("data/interim/entrega-02-corregida")
SALIDA = Path("dashboards/presentacion.html")

MESES = ["Dias4", "Dias5", "Dias6"]
UMBRALES = [1, 5, 10]
MIN_N_CICLO = 500          # n mínimo por ciclo y año para que el % sea creíble
TOP_CICLOS = 8             # cuántos ciclos mostrar (legibilidad en pantalla)

AZUL = "#2a78d6"
NARANJA = "#eb6834"
GRIS_25 = "#a9b4c2"
ROJO = "#d03b3b"
VERDE = "#2f8f6b"


def norm(v) -> str:
    if pd.isna(v):
        return ""
    s = str(v).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def cargar(pob: str) -> pd.DataFrame:
    """2025+2026 de una población, normalizado y con dias_total."""
    partes = []
    for anio in (2025, 2026):
        df = pd.read_parquet(CARPETA / f"{pob}_{anio}.parquet")
        df.columns = [norm(c).replace(" ", "_") for c in df.columns]
        df = df.drop_duplicates()
        if "ciclo" in df.columns:
            df["ciclo"] = df["ciclo"].map(norm)
        cols = [norm(c) for c in MESES]
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["dias_total"] = df[cols].sum(axis=1, min_count=1)
        df["anio"] = anio
        partes.append(df)
    return pd.concat(partes, ignore_index=True)


def tasa(df: pd.DataFrame, por, umbral: int) -> pd.DataFrame:
    d = df.copy()
    d["ok"] = d["dias_total"] >= umbral
    g = d.groupby(por, dropna=False).agg(n=("ok", "size"), acc=("ok", "sum")).reset_index()
    g["tasa"] = (100 * g["acc"] / g["n"]).round(1)
    return g


def estilo(fig, titulo, subtitulo, alto=460):
    fig.update_layout(
        title=dict(
            text=f"<b>{titulo}</b><br><span style='font-size:13px;color:#52514e'>{subtitulo}</span>",
            x=0.01, xanchor="left"),
        font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif",
                  size=14, color="#0b0b0b"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=""),
        margin=dict(t=100, l=70, r=30, b=60),
        bargap=0.28, bargroupgap=0.10, height=alto,
        plot_bgcolor="#fcfcfb", paper_bgcolor="#fcfcfb")
    fig.update_yaxes(gridcolor="#e1e0d9", zerolinecolor="#8f8e86", zerolinewidth=1.5)
    fig.update_xaxes(showgrid=False)
    return fig


# ============================================================
# LÁMINA 1 — el titular
# ============================================================
def lamina_umbral(datos):
    filas = []
    for pob, df in datos.items():
        for u in UMBRALES:
            t = tasa(df, ["anio"], u).set_index("anio")["tasa"]
            filas.append({"poblacion": pob.capitalize(),
                          "umbral": f"≥ {u} día{'s' if u > 1 else ''}",
                          "var_pp": round(t[2026] - t[2025], 1),
                          "t25": t[2025], "t26": t[2026]})
    d = pd.DataFrame(filas)

    fig = px.bar(d, x="umbral", y="var_pp", color="poblacion", barmode="group",
                 text="var_pp", custom_data=["poblacion", "t25", "t26"],
                 category_orders={"umbral": [f"≥ {u} día{'s' if u > 1 else ''}"
                                             for u in UMBRALES]},
                 color_discrete_map={"Estudiantes": AZUL, "Docentes": NARANJA})
    fig.update_traces(texttemplate="%{text:.1f} pp", textposition="outside",
                      hovertemplate="%{customdata[0]} · %{x}<br>"
                                    "%{customdata[1]:.1f}% → %{customdata[2]:.1f}%"
                                    "<br><b>%{y:.1f} pp</b><extra></extra>")
    estilo(fig, "1 · La caída se cuadruplica al exigir uso sostenido",
           "Variación 2025→2026 de la tasa, según cuántos días de acceso se exijan.")
    fig.update_yaxes(title="Cambio 2025 → 2026", ticksuffix=" pp")
    fig.update_xaxes(title="Umbral de días en el trimestre")

    est = d[d["poblacion"] == "Estudiantes"].set_index("umbral")["var_pp"]
    factor = abs(est.iloc[-1] / est.iloc[0])

    texto = (
        "Todos los análisis del reto miden lo mismo: <b>si el usuario entró</b>. Con esa "
        "definición la caída parece menor (−1,7 pp en estudiantes). Pero si en vez de "
        "preguntar <i>¿entró?</i> preguntamos <i>¿entró seguido?</i>, la caída "
        f"<b>se multiplica por {factor:.0f}</b>: llega a −6,8 pp exigiendo 10 días. "
        "En docentes pasa lo mismo (−3,1 → −6,4).<br><br>"
        "<b>El acceso casi no se movió. Lo que se rompió es la frecuencia de uso.</b> "
        "Medir solo 'accedió al menos una vez' subestima el problema por cuatro.")
    return "No es que entren menos. Es que entran menos veces.", texto, [fig]


# ============================================================
# LÁMINA 2 — el panel: blinda contra "es rotación de matrícula"
# ============================================================
def panel(df: pd.DataFrame):
    """Una fila por persona presente en ambos años, con días de cada año."""
    g = (df.groupby(["id_persona", "anio"])["dias_total"].max()
         .unstack())
    g = g.dropna(subset=[2025, 2026])
    g.columns = ["d25", "d26"]
    return g


def clasificar(g: pd.DataFrame) -> pd.Series:
    cond = [
        (g["d25"] == 0) & (g["d26"] == 0),
        (g["d25"] > 0) & (g["d26"] == 0),
        (g["d25"] == 0) & (g["d26"] > 0),
        g["d26"] < g["d25"],
        g["d26"] > g["d25"],
    ]
    etiquetas = ["Nunca entró", "Dejó de entrar", "Empezó a entrar",
                 "Bajó intensidad", "Subió intensidad"]
    out = pd.Series("Se mantuvo", index=g.index)
    for c, e in zip(cond, etiquetas):
        out = out.mask(c & (out == "Se mantuvo"), e)
    return out


def lamina_panel(datos):
    filas, resumen = [], {}
    for pob, df in datos.items():
        g = panel(df)
        g["cat"] = clasificar(g)
        vc = g["cat"].value_counts(normalize=True).mul(100).round(1)
        for cat, pct in vc.items():
            filas.append({"poblacion": pob.capitalize(), "cat": cat, "pct": pct,
                          "n": int(g["cat"].eq(cat).sum())})
        resumen[pob] = dict(n=len(g), d25=round(g["d25"].mean(), 1),
                            d26=round(g["d26"].mean(), 1),
                            bajo=vc.get("Bajó intensidad", 0),
                            subio=vc.get("Subió intensidad", 0))
    d = pd.DataFrame(filas)

    orden = ["Bajó intensidad", "Se mantuvo", "Subió intensidad",
             "Dejó de entrar", "Empezó a entrar", "Nunca entró"]
    fig = px.bar(d, x="cat", y="pct", color="poblacion", barmode="group",
                 text="pct", custom_data=["poblacion", "n"],
                 category_orders={"cat": orden},
                 color_discrete_map={"Estudiantes": AZUL, "Docentes": NARANJA})
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                      hovertemplate="%{customdata[0]} · %{x}<br>%{y:.1f}%"
                                    "<br>%{customdata[1]:,} personas<extra></extra>")
    estilo(fig, "2 · Qué pasó con las MISMAS personas entre 2025 y 2026",
           "Solo usuarios presentes en ambos años. Sin altas ni bajas de matrícula.")
    fig.update_yaxes(title="% del panel", ticksuffix="%")
    fig.update_xaxes(title="")

    # anio va como TEXTO a propósito: si va como número, plotly lo toma como
    # escala continua y el color_discrete_map no se aplica.
    dias = pd.DataFrame([
        {"poblacion": p.capitalize(), "anio": str(a), "dias": r[f"d{a % 100}"]}
        for p, r in resumen.items() for a in (2025, 2026)])
    fig2 = px.bar(dias, x="poblacion", y="dias", color="anio", barmode="group",
                  text="dias", color_discrete_map={"2025": GRIS_25, "2026": AZUL})
    fig2.update_traces(texttemplate="%{text:.1f}", textposition="outside",
                       hovertemplate="%{x} · %{fullData.name}"
                                     "<br>%{y:.1f} días<extra></extra>")
    estilo(fig2, "2b · Días de uso promedio, mismas personas",
           "Y estas personas están un año MÁS GRANDES en 2026.", alto=380)
    fig2.update_yaxes(title="Días en el trimestre")
    fig2.update_xaxes(title="")

    e, dc = resumen["estudiantes"], resumen["docentes"]
    texto = (
        f"La objeción natural es que la caída sea rotación de matrícula: entra gente nueva "
        f"que usa menos y eso arrastra el promedio. Para descartarlo miramos únicamente a "
        f"las personas presentes en <b>los dos años</b> ({e['n']:,} estudiantes y "
        f"{dc['n']:,} docentes).<br><br>"
        f"Ahí también cae: <b>{e['bajo']:.1f}% de los estudiantes bajó su intensidad</b> "
        f"frente a {e['subio']:.1f}% que la subió, y el promedio pasa de {e['d25']} a "
        f"{e['d26']} días. En docentes, de {dc['d25']} a {dc['d26']}.<br><br>"
        f"<b>Y hay un agravante:</b> en 2026 estos estudiantes tienen un año más, y el uso "
        f"de CREA crece con el grado. Deberían haber subido. Bajaron. "
        f"<b>El desenganche real es mayor que el que muestra el número.</b>")
    return "Entre las mismas personas, también cae", texto, [fig, fig2]


# ============================================================
# LÁMINA 3 — dónde ocurre el desenganche
# ============================================================
def tabla_ciclo(df: pd.DataFrame) -> pd.DataFrame:
    """Por ciclo: var_pp con umbral 1 y 10, más volumen de usuarios perdidos."""
    out = {}
    for u in (1, 10):
        t = tasa(df, ["ciclo", "anio"], u)
        piv = t.pivot_table(index="ciclo", columns="anio", values=["n", "tasa"])
        piv.columns = [f"{a}_{b}" for a, b in piv.columns]
        piv = piv[(piv["n_2025"].fillna(0) >= MIN_N_CICLO) &
                  (piv["n_2026"].fillna(0) >= MIN_N_CICLO)]
        out[u] = piv

    g = pd.DataFrame({
        "var_pp_1": out[1]["tasa_2026"] - out[1]["tasa_2025"],
        "var_pp_10": out[10]["tasa_2026"] - out[10]["tasa_2025"],
        "n_2026": out[10]["n_2026"],
        "t25": out[10]["tasa_2025"],
        "t26": out[10]["tasa_2026"],
    }).dropna()
    g["profundizacion"] = (g["var_pp_10"] - g["var_pp_1"]).round(1)
    g["perdidos"] = (g["n_2026"] * (g["t25"] - g["t26"]) / 100).round(0)
    g["rel"] = (100 * (g["t26"] - g["t25"]) / g["t25"]).round(1)
    for c in ("var_pp_1", "var_pp_10"):
        g[c] = g[c].round(1)
    return g.reset_index().assign(ciclo=lambda x: x["ciclo"].str.title())


def lamina_ciclo(tab: pd.DataFrame):
    sub = tab.nsmallest(TOP_CICLOS, "profundizacion")
    largo = sub.melt(id_vars="ciclo", value_vars=["var_pp_1", "var_pp_10"],
                     var_name="metrica", value_name="var_pp")
    largo["metrica"] = largo["metrica"].map({"var_pp_1": "Acceso (≥ 1 día)",
                                             "var_pp_10": "Uso sostenido (≥ 10 días)"})
    orden = sub.sort_values("var_pp_10")["ciclo"].tolist()

    fig = px.bar(largo, x="var_pp", y="ciclo", color="metrica", barmode="group",
                 orientation="h", text="var_pp",
                 category_orders={"ciclo": orden},
                 color_discrete_map={"Acceso (≥ 1 día)": GRIS_25,
                                     "Uso sostenido (≥ 10 días)": ROJO})
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside",
                      hovertemplate="%{y}<br>%{fullData.name}: %{x:.1f} pp<extra></extra>")
    estilo(fig, "3 · Los ciclos donde el desenganche es más específico",
           "Gris: cambio en acceso. Rojo: cambio en uso sostenido. La distancia entre "
           "ambos es lo que no se ve midiendo solo acceso.", alto=520)
    fig.update_xaxes(title="Cambio 2025 → 2026", ticksuffix=" pp")
    fig.update_yaxes(title="")

    top = sub.nsmallest(3, "profundizacion")
    lista = ", ".join(f"<b>{r['ciclo']}</b> ({r['var_pp_1']:.1f} → {r['var_pp_10']:.1f} pp)"
                      for _, r in top.iterrows())
    texto = (
        f"El desenganche se concentra en los ciclos del <b>núcleo obligatorio</b>: {lista}.<br><br>"
        "Estos ciclos tienen acceso prácticamente universal (94–96% entra al menos una vez) "
        "y ese acceso <b>casi no se movió</b>. Lo que se derrumbó es la frecuencia: caen entre "
        "9 y 10 puntos en uso sostenido.<br><br>"
        "<b>No perdieron usuarios. Perdieron la rutina.</b> Siguen entrando a CREA, pero "
        "dejó de ser parte del día a día.")
    return "Se soltó el núcleo obligatorio", texto, [fig]


# ============================================================
# LÁMINA 4 — volumen, no porcentaje
# ============================================================
def lamina_volumen(tab: pd.DataFrame):
    sub = tab.nlargest(TOP_CICLOS, "perdidos").sort_values("perdidos", ascending=False)

    fig = px.bar(sub, x="ciclo", y="perdidos", text="perdidos",
                 color_discrete_sequence=[VERDE],
                 custom_data=["rel", "t25", "t26", "n_2026"])
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside",
                      hovertemplate="<b>%{x}</b><br>%{customdata[3]:,} alumnos en 2026"
                                    "<br>%{customdata[1]:.1f}% → %{customdata[2]:.1f}%"
                                    " (%{customdata[0]:.1f}% relativo)"
                                    "<br><b>%{y:,.0f} alumnos menos</b><extra></extra>")
    estilo(fig, "4 · Dónde una intervención alcanza a más estudiantes",
           "Alumnos que dejaron de alcanzar 10+ días de uso, a matrícula 2026 fija.", alto=480)
    fig.update_yaxes(title="Alumnos afectados")
    fig.update_xaxes(title="", tickangle=-30)

    peor_rel = tab.nsmallest(1, "rel").iloc[0]
    top3 = tab.nlargest(3, "perdidos")
    total = tab[tab["perdidos"] > 0]["perdidos"].sum()
    concentra = 100 * top3["perdidos"].sum() / total

    texto = (
        f"El ranking por porcentaje y el ranking por cantidad de gente <b>no coinciden</b>, y "
        f"para decidir dónde intervenir importa el segundo.<br><br>"
        f"La peor caída relativa es <b>{peor_rel['ciclo']}</b> ({peor_rel['rel']:.1f}%), pero "
        f"son <b>{peor_rel['perdidos']:,.0f} alumnos</b>. En volumen, el problema está en "
        + ", ".join(f"<b>{r['ciclo']}</b> ({r['perdidos']:,.0f})" for _, r in top3.iterrows())
        + f": entre los tres explican el <b>{concentra:.0f}%</b> de los {total:,.0f} alumnos "
        f"que dejaron el uso sostenido.<br><br>"
        f"<b>Recomendación:</b> priorizar el núcleo obligatorio, donde el acceso ya está "
        f"resuelto y lo que falta es reconstruir la frecuencia de uso.")
    return "Dónde poner los recursos", texto, [fig]


# ============================================================
# ARMADO
# ============================================================
PLANTILLA = """<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CREA 2025 vs 2026 — Intensidad de uso</title>
<style>
  body {{ font-family: system-ui, -apple-system, "Segoe UI", sans-serif; color: #0b0b0b;
         background: #f4f4f1; margin: 0; line-height: 1.65; }}
  .wrap {{ max-width: 1080px; margin: 0 auto; padding: 44px 24px 90px; }}
  header {{ margin-bottom: 34px; }}
  header h1 {{ font-size: 34px; margin: 0 0 8px; letter-spacing: -0.5px; }}
  header p {{ color: #52514e; margin: 0; font-size: 16px; }}
  header .meta {{ color: #898781; font-size: 13px; margin-top: 10px; }}
  section {{ background: #fff; border: 1px solid rgba(11,11,11,.09); border-radius: 14px;
            padding: 30px 32px; margin: 26px 0; box-shadow: 0 1px 3px rgba(0,0,0,.04); }}
  section h2 {{ font-size: 25px; margin: 0 0 16px; letter-spacing: -0.3px; }}
  .num {{ display: inline-block; background: #0b0b0b; color: #fff; border-radius: 8px;
          width: 30px; height: 30px; line-height: 30px; text-align: center;
          font-size: 16px; margin-right: 10px; vertical-align: 3px; }}
  .analisis {{ font-size: 16px; color: #2a2a28; margin: 0 0 22px; }}
  .grafico {{ margin: 10px 0; }}
  footer {{ color: #898781; font-size: 13px; text-align: center; margin-top: 44px; }}
</style></head>
<body><div class="wrap">
<header>
  <h1>El acceso a CREA casi no cayó. La frecuencia de uso, sí.</h1>
  <p>Análisis de la baja de uso entre abril–junio de 2025 y 2026.</p>
  <div class="meta">Presentación a Ceibal · 20 de julio de 2026 · datos anonimizados ANEP/Ceibal</div>
</header>
{secciones}
<footer>Generado con presentacion.py</footer>
</div></body></html>"""


def main():
    print("Cargando datos...")
    datos = {p: cargar(p) for p in ("estudiantes", "docentes")}
    for p, df in datos.items():
        print(f"  {p}: {len(df):,} filas")

    print("Calculando por ciclo...")
    tab = tabla_ciclo(datos["estudiantes"])
    print(f"  {len(tab)} ciclos comparables (n >= {MIN_N_CICLO} en ambos años)")

    print("Armando láminas...")
    secciones = [
        lamina_umbral(datos),
        lamina_panel(datos),
        lamina_ciclo(tab),
        lamina_volumen(tab),
    ]

    bloques, primero = [], True
    for i, (titulo, texto, figuras) in enumerate(secciones, start=1):
        graficos = ""
        for fig in figuras:
            graficos += ('<div class="grafico">' + fig.to_html(
                full_html=False,
                include_plotlyjs="cdn" if primero else False) + "</div>")
            primero = False
        bloques.append(
            f'<section><h2><span class="num">{i}</span>{titulo}</h2>'
            f'<p class="analisis">{texto}</p>{graficos}</section>')

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text(PLANTILLA.format(secciones="\n".join(bloques)), encoding="utf-8")
    print("\n  ->", SALIDA.resolve())
    print("  Abrilo con:  open dashboards/presentacion.html")


if __name__ == "__main__":
    main()