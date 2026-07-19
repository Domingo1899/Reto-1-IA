"""
visualizaciones.py — Reporte único de análisis (2025 vs 2026).

Genera UN solo archivo HTML legible como documento (dashboards/reporte.html),
donde cada sección combina un texto de análisis + un gráfico plotly interactivo,
en el orden de la narrativa con drill-down:

  1. Panorama general país
  2. Por quintil socioeconómico (el eje del análisis)
  3. Quintil 5 x grado (drill-down)
  4. Mapa coroplético por departamento

La idea es que se lea como un informe, no como gráficos sueltos. El código está
pensado para ser fácil de leer y modificar: una función corta por sección, cada
una devuelve (titulo, texto, figura), y plotly.express directo. Se repite algo
de estilo a propósito, para que cada sección sea autónoma.

Notas de datos (ver contexto_socioeconomico.py y graficos):
- El quintil sale de CONTEXTO, poblado SOLO para primaria (DGEIP). Por eso el
  análisis por quintil es de primaria, y cubre ~44% de los estudiantes.
- El hallazgo "el quintil 5 cae más" solo aparece con uso sostenido (>=10 días),
  no con acceso básico (>=1 día). Por eso graficamos ambas métricas.
- Nombres de departamento: el GeoJSON usa NAME_1 (Title Case con tildes) y
  nuestros datos usan MAYÚSCULAS sin tildes; normalizando coinciden 19/19.
"""

import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
import plotly.express as px

CARPETA = Path("data/interim/entrega-02-corregida")
METRICAS = Path("data/processed/metricas")
GEOJSON = Path("data/mapa/uruguay.json")
SALIDA = Path("dashboards/reporte.html")
MESES = ["Dias4", "Dias5", "Dias6"]
EXCL = {"", "sin clasificar"}  # valores de CONTEXTO que se excluyen

# Paleta validada con el validador del skill dataviz (par CVD-safe claro/oscuro).
AZUL = "#2a78d6"      # urbano / año 2026 (foco)
NARANJA = "#eb6834"   # rural
GRIS_25 = "#a9b4c2"   # año 2025 (recesivo, "el pasado")
ROJO = "#d03b3b"      # resaltar caída / quintil 5

GRADOS = {"1o": "1º", "2o": "2º", "3o": "3º", "4o": "4º", "5o": "5º", "6o": "6º"}


def norm(v) -> str:
    if pd.isna(v):
        return ""
    s = str(v).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def cargar(pob: str) -> pd.DataFrame:
    """Carga 2025+2026 de una población con columnas de quintil y días."""
    partes = []
    for anio in (2025, 2026):
        df = pd.read_parquet(CARPETA / f"{pob}_{anio}.parquet").drop_duplicates()
        df["ctx"] = df["CONTEXTO"].map(norm)
        for c in MESES:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["dias_total"] = df[MESES].sum(axis=1, min_count=1)
        df["anio"] = anio
        partes.append(df)
    return pd.concat(partes, ignore_index=True)


def tasa(df: pd.DataFrame, por, umbral: int) -> pd.DataFrame:
    """% que accedió (>= umbral días), agrupado por las columnas 'por'."""
    df = df.copy()
    df["ok"] = df["dias_total"] >= umbral
    g = df.groupby(por, dropna=False).agg(n=("ok", "size"), acc=("ok", "sum")).reset_index()
    g["tasa"] = (100 * g["acc"] / g["n"]).round(1)
    return g


def solo_quintil(df: pd.DataFrame) -> pd.DataFrame:
    """Deja registros con quintil válido y agrega columnas q (1-5) y zona."""
    d = df[~df["ctx"].isin(EXCL)].copy()
    d["q"] = d["ctx"].map(lambda v: int(re.search(r"(\d)", v).group()))
    d["zona"] = d["ctx"].map(lambda v: "Urbano" if "urbano" in v else "Rural")
    return d


def estilo(fig, titulo, subtitulo):
    """Estilo común para todas las figuras (light, limpio)."""
    fig.update_layout(
        title=dict(
            text=f"<b>{titulo}</b><br><span style='font-size:13px;color:#52514e'>{subtitulo}</span>",
            x=0.01, xanchor="left",
        ),
        font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", size=13, color="#0b0b0b"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=""),
        margin=dict(t=95, l=60, r=30, b=55),
        bargap=0.25, bargroupgap=0.08,
        plot_bgcolor="#fcfcfb", paper_bgcolor="#fcfcfb",
    )
    fig.update_yaxes(gridcolor="#e1e0d9", zerolinecolor="#c3c2b7")
    fig.update_xaxes(showgrid=False)
    return fig


# ============================================================
# 1. PANORAMA GENERAL PAÍS
# ============================================================
def seccion_panorama():
    filas = []
    for pob in ("estudiantes", "docentes"):
        g = tasa(cargar(pob), ["anio"], 1)
        for _, r in g.iterrows():
            filas.append({"poblacion": pob.capitalize(), "anio": str(r["anio"]),
                          "tasa": r["tasa"], "acc": int(r["acc"])})
    d = pd.DataFrame(filas)

    fig = px.bar(d, x="poblacion", y="tasa", color="anio", barmode="group",
                 text="tasa", color_discrete_map={"2025": GRIS_25, "2026": AZUL},
                 custom_data=["anio", "acc"])
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                      hovertemplate="%{x} · %{customdata[0]}<br>Tasa de acceso: %{y:.1f}%"
                                    "<br>Accedieron: %{customdata[1]:,}<extra></extra>")
    estilo(fig, "1 · Panorama país: tasa de acceso a CREA",
           "% que accedió al menos 1 día (abril–junio de cada año).")
    fig.update_yaxes(range=[0, 100], title="Tasa de acceso", ticksuffix="%")
    fig.update_xaxes(title="")

    texto = (
        "A nivel país la caída existe pero es <b>leve</b>: los estudiantes pasan de 81.6% "
        "a 79.9% (−1.7 pp) y los docentes de 82.9% a 79.8% (−3.1 pp). Una caída global tan "
        "chica plantea la pregunta central del reto: ¿hay menos gente (cambio de población) "
        "o la misma gente entra menos (cambio de comportamiento)? El promedio país esconde "
        "las diferencias reales, que recién aparecen al desagregar por contexto socioeconómico.")
    return "Panorama general país", texto, [fig]


# ============================================================
# 2. POR QUINTIL SOCIOECONÓMICO
# ============================================================
def seccion_quintil():
    todo = cargar("estudiantes")
    d = solo_quintil(todo)
    cobertura = 100 * len(d) / len(todo)

    figuras = []
    for archivo, etiqueta, umbral in [("≥ 1 día", "Acceso (≥ 1 día)", 1),
                                      ("≥ 10 días", "Uso sostenido (≥ 10 días)", 10)]:
        g = tasa(d, ["q", "zona", "anio"], umbral)
        g["quintil"] = "Q" + g["q"].astype(str)
        g["anio"] = g["anio"].astype(str)
        fig = px.bar(g, x="quintil", y="tasa", color="zona", pattern_shape="anio",
                     barmode="group",
                     category_orders={"quintil": [f"Q{i}" for i in range(1, 6)],
                                      "anio": ["2025", "2026"]},
                     color_discrete_map={"Urbano": AZUL, "Rural": NARANJA},
                     pattern_shape_map={"2025": "/", "2026": ""},
                     custom_data=["zona", "anio", "n"])
        fig.update_traces(marker_line_width=0.5, marker_line_color="#fcfcfb",
                          hovertemplate="Quintil %{x} · %{customdata[0]} · %{customdata[1]}"
                                        "<br>Tasa: %{y:.1f}%<br>n = %{customdata[2]:,}<extra></extra>")
        fig.add_vrect(x0=3.5, x1=4.5, fillcolor=ROJO, opacity=0.06, line_width=0, layer="below")
        estilo(fig, f"2 · Tasa por quintil socioeconómico — {etiqueta}",
               "Rayado = 2025, sólido = 2026. Color = zona. Quintil 1 = más vulnerable, 5 = más favorecido.")
        fig.update_yaxes(range=[0, 100], title="Tasa", ticksuffix="%")
        fig.update_xaxes(title="Quintil del centro (1 → 5)")
        figuras.append(fig)

    # Punchline: la caída (var_pp) por quintil, ambas métricas juntas
    filas = []
    for etiqueta, umbral in [("Acceso (≥ 1 día)", 1), ("Uso sostenido (≥ 10 días)", 10)]:
        piv = tasa(d, ["q", "anio"], umbral).pivot_table(index="q", columns="anio", values="tasa")
        for q, r in piv.iterrows():
            filas.append({"quintil": f"Q{q}", "q": q, "metrica": etiqueta,
                          "var_pp": round(r[2026] - r[2025], 1)})
    dp = pd.DataFrame(filas).sort_values("q")
    fig3 = px.bar(dp, x="quintil", y="var_pp", color="metrica", barmode="group", text="var_pp",
                  category_orders={"quintil": [f"Q{i}" for i in range(1, 6)]},
                  color_discrete_map={"Acceso (≥ 1 día)": GRIS_25, "Uso sostenido (≥ 10 días)": ROJO})
    fig3.update_traces(texttemplate="%{text:.1f}", textposition="outside",
                       hovertemplate="Quintil %{x}<br>%{fullData.name}<br>Cambio: %{y:.1f} pp<extra></extra>")
    fig3.add_annotation(x="Q5", y=dp["var_pp"].min(), yshift=-16, showarrow=False,
                        text="<b>El quintil 5 es el que MÁS cae</b>", font=dict(color=ROJO, size=12))
    estilo(fig3, "2 · El hallazgo: la mayor caída está en el quintil 5",
           "Cambio 2025→2026 en puntos porcentuales, por métrica de acceso.")
    fig3.update_yaxes(title="Cambio 2025 → 2026", ticksuffix=" pp")
    fig3.update_xaxes(title="Quintil del centro (1 → 5)")

    texto = (
        f"<b>Cobertura:</b> el contexto socioeconómico (columna CONTEXTO) está poblado solo "
        f"para primaria (DGEIP), así que estos gráficos cubren el <b>{cobertura:.0f}%</b> de los "
        f"estudiantes (el resto —secundaria y UTU— no trae quintil comparable entre años). "
        f"<b>Por qué dos métricas:</b> con acceso básico (≥1 día) la caída es pareja entre "
        f"quintiles y el efecto no se ve; el hallazgo aparece recién con <b>uso sostenido</b> "
        f"(≥10 días). Ahí surge el resultado contraintuitivo: el <b>quintil 5</b>, el de mejor "
        f"posición socioeconómica, es el que <b>más cae</b> (−9.3 pp urbano), por encima de los "
        f"quintiles más vulnerables. El uso sostenido —no el acceso mínimo— es donde se rompió algo en 2026.")
    return "Por quintil socioeconómico", texto, figuras + [fig3]


# ============================================================
# 3. QUINTIL 5 x GRADO (drill-down)
# ============================================================
def seccion_quintil_grado():
    d = solo_quintil(cargar("estudiantes"))
    d["grado"] = d["grado"].astype(str).map(norm).map(GRADOS)
    d = d[(d["q"] == 5) & d["grado"].isin(GRADOS.values())]

    filas = []
    for etiqueta, umbral in [("Acceso (≥ 1 día)", 1), ("Uso sostenido (≥ 10 días)", 10)]:
        g = tasa(d, ["grado", "anio"], umbral)
        g["metrica"] = etiqueta
        filas.append(g)
    g = pd.concat(filas, ignore_index=True)
    g["anio"] = g["anio"].astype(str)

    fig = px.bar(g, x="grado", y="tasa", color="anio", barmode="group", facet_col="metrica",
                 text="tasa", category_orders={"grado": list(GRADOS.values()),
                 "anio": ["2025", "2026"], "metrica": ["Acceso (≥ 1 día)", "Uso sostenido (≥ 10 días)"]},
                 color_discrete_map={"2025": GRIS_25, "2026": AZUL}, custom_data=["anio", "n"])
    fig.update_traces(texttemplate="%{text:.0f}", textposition="outside",
                      hovertemplate="%{x} · %{customdata[0]}<br>Tasa: %{y:.1f}%<br>n = %{customdata[1]:,}<extra></extra>")
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    estilo(fig, "3 · Drill-down del quintil 5: ¿en qué grado se concentra la caída?",
           "Solo primaria, quintil 5. Tasa por grado, 2025 vs 2026.")
    fig.update_yaxes(range=[0, 108], title="Tasa", ticksuffix="%")
    fig.update_xaxes(title="")

    texto = (
        "Dentro del quintil 5, la caída no es uniforme entre grados. Con uso sostenido "
        "(≥10 días) se concentra fuerte en <b>3º y 4º grado</b> (−15.2 y −14.5 pp), mientras "
        "que 5º y 6º caen bastante menos. Es un patrón muy localizado: sugiere que el problema "
        "de 2026 golpeó a un tramo específico de la primaria favorecida, no a toda por igual.")
    return "Quintil 5 × grado (drill-down)", texto, [fig]


# ============================================================
# 4. MAPA COROPLÉTICO POR DEPARTAMENTO
# ============================================================
def topojson_a_geojson(topo, objeto):
    """Convierte un TopoJSON a GeoJSON FeatureCollection (arcos + cuantización)."""
    esc, tr = topo["transform"]["scale"], topo["transform"]["translate"]
    def decodificar(arco):
        pts, x, y = [], 0, 0
        for dx, dy in arco:
            x += dx; y += dy
            pts.append([x * esc[0] + tr[0], y * esc[1] + tr[1]])
        return pts
    arcos = [decodificar(a) for a in topo["arcs"]]
    def coords(i):
        return arcos[i] if i >= 0 else arcos[~i][::-1]
    def anillo(indices):
        linea = []
        for j, i in enumerate(indices):
            seg = coords(i)
            linea.extend(seg if j == 0 else seg[1:])
        return linea
    feats = []
    for g in topo["objects"][objeto]["geometries"]:
        if g["type"] == "Polygon":
            geom = [anillo(r) for r in g["arcs"]]
        else:  # MultiPolygon
            geom = [[anillo(r) for r in poly] for poly in g["arcs"]]
        feats.append({"type": "Feature", "properties": g["properties"],
                      "geometry": {"type": g["type"], "coordinates": geom}})
    return {"type": "FeatureCollection", "features": feats}


def seccion_mapa():
    topo = json.loads(GEOJSON.read_text())
    geo = topojson_a_geojson(topo, "uruguay")
    # mapa norm(NAME_1) -> NAME_1, para traducir nuestros dept_nombre al del GeoJSON
    a_geo = {norm(f["properties"]["NAME_1"]): f["properties"]["NAME_1"] for f in geo["features"]}

    d = pd.read_csv(METRICAS / "comparar_dept_nombre_estudiantes.csv", sep=";", decimal=",")
    d["depto"] = d["dept_nombre"].map(a_geo)

    lim = max(abs(d["var_tasa_pp"].min()), abs(d["var_tasa_pp"].max()))
    fig = px.choropleth(d, geojson=geo, locations="depto", featureidkey="properties.NAME_1",
                        color="var_tasa_pp", color_continuous_scale="RdBu",
                        range_color=[-lim, lim], custom_data=["dept_nombre", "tasa_%_2025", "tasa_%_2026"])
    fig.update_traces(marker_line_color="#fcfcfb", marker_line_width=0.6,
                      hovertemplate="<b>%{customdata[0]}</b><br>2025: %{customdata[1]:.1f}% → "
                                    "2026: %{customdata[2]:.1f}%<br>Cambio: %{z:.1f} pp<extra></extra>")
    fig.update_geos(fitbounds="locations", visible=False)
    estilo(fig, "4 · Caída de la tasa de acceso por departamento",
           "Variación 2025→2026 en puntos porcentuales (estudiantes, acceso ≥1 día). Rojo = más caída.")
    fig.update_layout(coloraxis_colorbar=dict(title="pp", ticksuffix=" pp"),
                      margin=dict(t=95, l=10, r=10, b=10))

    peor = d.loc[d["var_tasa_pp"].idxmin()]
    texto = (
        "El mapa muestra que la caída no es homogénea en el territorio: se concentra en algunos "
        f"departamentos (en rojo más intenso) mientras otros casi no cambian. El mayor retroceso "
        f"está en <b>{peor['dept_nombre'].title()}</b> ({peor['var_tasa_pp']:.1f} pp). "
        "Esta vista país-level es el puente hacia el último paso del análisis: elegir un "
        "departamento vulnerable y hacer foco en él como cierre.")
    return "Mapa por departamento", texto, [fig]
"""
seccion_ciclo.py — CÓDIGO PARA PEGAR DENTRO DE visualizaciones.py

CÓMO INSTALARLO (2 pasos):

1. Copiá la función seccion_ciclo() de abajo y pegala en visualizaciones.py,
   justo ANTES del bloque "ARMADO DEL REPORTE" (donde está PLANTILLA).

2. En main(), agregala a la lista de secciones:

       secciones = [seccion_panorama(), seccion_quintil(),
                    seccion_quintil_grado(),
                    seccion_ciclo("estudiantes"),      # <-- NUEVO
                    seccion_ciclo("docentes"),         # <-- NUEVO
                    seccion_mapa()]

Requisito: haber corrido antes ciclo_estudiantes.py y ciclo_docentes.py, que
son los que generan los CSV que esta sección lee.


QUÉ GRAFICA Y POR QUÉ
Tres figuras, en orden de argumento:

  A. PROFUNDIZACIÓN — var_pp con umbral 1 vs umbral 10, por ciclo. Es el mismo
     hallazgo central del reto (no cae el acceso, cae la intensidad) pero
     mostrado ciclo por ciclo. Las barras rojas siempre más largas que las
     grises = el patrón se repite en todos lados, no es un artefacto.

  B. VOLUMEN — cuántos usuarios MENOS alcanzaron uso sostenido. Un % grande
     sobre 800 alumnos pesa menos que un % moderado sobre 223.000. Esta es la
     figura que responde "¿dónde poner los recursos?".

     Definición: usuarios_perdidos = n_2026 * (tasa_2025 - tasa_2026) / 100
     Es decir: cuántos alumnos menos llegaron a >=10 días en 2026 respecto de
     los que habrían llegado si la tasa de 2025 se hubiera mantenido, usando la
     matrícula de 2026. Al fijar la matrícula, aísla el cambio de COMPORTAMIENTO
     del cambio de POBLACIÓN (la misma distinción de metricas_acceso.py).

  C. LAS DOS COSAS JUNTAS — dispersión % de caída vs cantidad de gente, tamaño
     de burbuja = matrícula. Arriba a la derecha están los casos que importan
     de verdad (cae mucho Y afecta a muchos). Sirve para mostrar de un golpe
     que el ranking porcentual solo es engañoso.
"""

# ============================================================
# 5. POR CICLO (pegar en visualizaciones.py)
# ============================================================
def seccion_ciclo(pob="estudiantes"):
    """Caída por ciclo: profundización + volumen absoluto de usuarios."""
    d = pd.read_csv(METRICAS / f"ciclo_{pob}.csv", sep=";", decimal=",")
    d["ciclo"] = d["ciclo"].str.title()

    u1 = d[d["umbral"] == 1].set_index("ciclo")
    u10 = d[d["umbral"] == 10].set_index("ciclo")

    # Usuarios que dejaron de usar CREA de forma sostenida, a matrícula 2026.
    # Fija la población para que el número sea comportamiento, no matrícula.
    g = pd.DataFrame({
        "var_pp_1": u1["var_pp"],
        "var_pp_10": u10["var_pp"],
        "n_2026": u10["n_2026"],
        "tasa_2025": u10["tasa_2025"],
        "tasa_2026": u10["tasa_2026"],
    }).dropna().reset_index()
    g["perdidos"] = (g["n_2026"] * (g["tasa_2025"] - g["tasa_2026"]) / 100).round(0)
    g["caida_rel"] = (100 * (g["tasa_2026"] - g["tasa_2025"]) / g["tasa_2025"]).round(1)

    etiqueta = "alumnos" if pob == "estudiantes" else "docentes"

    # ---- A. profundización: acceso vs uso sostenido ---------------------
    largo = g.melt(id_vars="ciclo", value_vars=["var_pp_1", "var_pp_10"],
                   var_name="metrica", value_name="var_pp")
    largo["metrica"] = largo["metrica"].map({"var_pp_1": "Acceso (≥ 1 día)",
                                             "var_pp_10": "Uso sostenido (≥ 10 días)"})
    orden = g.sort_values("var_pp_10")["ciclo"].tolist()

    figA = px.bar(largo, x="var_pp", y="ciclo", color="metrica", barmode="group",
                  orientation="h", text="var_pp",
                  category_orders={"ciclo": orden[::-1]},
                  color_discrete_map={"Acceso (≥ 1 día)": GRIS_25,
                                      "Uso sostenido (≥ 10 días)": ROJO})
    figA.update_traces(texttemplate="%{text:.1f}", textposition="outside",
                       hovertemplate="%{y}<br>%{fullData.name}: %{x:.1f} pp<extra></extra>")
    estilo(figA, f"5 · La caída por ciclo es de intensidad, no de acceso — {etiqueta.capitalize()}",
           "Cambio 2025→2026 en pp. La barra roja es casi siempre más larga: el uso sostenido cae más que el acceso.")
    figA.update_xaxes(title="Cambio 2025 → 2026", ticksuffix=" pp")
    figA.update_yaxes(title="")

    # ---- B. volumen absoluto -------------------------------------------
    gv = g.sort_values("perdidos", ascending=False)
    figB = px.bar(gv, x="ciclo", y="perdidos", text="perdidos",
                  color_discrete_sequence=[AZUL],
                  custom_data=["tasa_2025", "tasa_2026", "var_pp_10", "n_2026"])
    figB.update_traces(texttemplate="%{text:,.0f}", textposition="outside",
                       hovertemplate="<b>%{x}</b><br>%{customdata[3]:,} " + etiqueta +
                                     " en 2026<br>Tasa: %{customdata[0]:.1f}% → %{customdata[1]:.1f}%"
                                     " (%{customdata[2]:.1f} pp)<br><b>%{y:,.0f} " + etiqueta +
                                     " menos</b> con uso sostenido<extra></extra>")
    estilo(figB, f"5 · Dónde está el grueso del problema — {etiqueta.capitalize()}",
           f"{etiqueta.capitalize()} que dejaron de alcanzar 10+ días de uso, a matrícula 2026. "
           "Un % alto sobre pocos casos pesa poco; esta figura ordena por gente real.")
    figB.update_yaxes(title=f"{etiqueta.capitalize()} afectados")
    figB.update_xaxes(title="", tickangle=-35)

    # ---- C. % vs volumen ------------------------------------------------
    figC = px.scatter(g, x="caida_rel", y="perdidos", size="n_2026", text="ciclo",
                      size_max=55, color_discrete_sequence=[AZUL],
                      custom_data=["var_pp_10", "n_2026"])
    figC.update_traces(textposition="top center", textfont_size=10,
                       marker=dict(opacity=0.75, line=dict(width=1, color="#fcfcfb")),
                       hovertemplate="<b>%{text}</b><br>Caída relativa: %{x:.1f}%"
                                     "<br>%{y:,.0f} " + etiqueta + " afectados"
                                     "<br>Matrícula 2026: %{customdata[1]:,}<extra></extra>")
    estilo(figC, "5 · Caída porcentual vs cantidad de gente afectada",
           "Tamaño = matrícula 2026. Izquierda = cae más en %. Arriba = afecta a más gente. "
           "Los casos accionables están arriba, no a la izquierda.")
    figC.update_xaxes(title="Caída relativa del uso sostenido", ticksuffix="%")
    figC.update_yaxes(title=f"{etiqueta.capitalize()} afectados")

    # ---- texto ----------------------------------------------------------
    top = g.nlargest(3, "perdidos")
    peor_rel = g.nsmallest(1, "caida_rel").iloc[0]
    total = g["perdidos"].sum()
    concentra = 100 * top["perdidos"].sum() / total

    texto = (
        f"El corte por ciclo tiene una trampa: el ranking por porcentaje y el ranking por "
        f"cantidad de gente <b>no coinciden</b>. El ciclo con mayor caída relativa es "
        f"<b>{peor_rel['ciclo']}</b> ({peor_rel['caida_rel']:.1f}%), pero involucra apenas "
        f"{peor_rel['perdidos']:,.0f} {etiqueta}. En volumen, el problema está concentrado en "
        f"<b>{top.iloc[0]['ciclo']}</b> ({top.iloc[0]['perdidos']:,.0f} {etiqueta}), "
        f"<b>{top.iloc[1]['ciclo']}</b> ({top.iloc[1]['perdidos']:,.0f}) y "
        f"<b>{top.iloc[2]['ciclo']}</b> ({top.iloc[2]['perdidos']:,.0f}): entre los tres "
        f"explican el <b>{concentra:.0f}%</b> de los {total:,.0f} {etiqueta} que dejaron de "
        f"usar CREA de forma sostenida. Las dos lecturas son válidas y responden preguntas "
        f"distintas: la relativa dice dónde el deterioro fue más severo, la absoluta dice "
        f"dónde una intervención alcanza a más gente."
    )
    return f"Por ciclo — {etiqueta}", texto, [figA, figB, figC]


# ============================================================
# ARMADO DEL REPORTE
# ============================================================
PLANTILLA = """<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CREA 2025 vs 2026 — Análisis con drill-down</title>
<style>
  body {{ font-family: system-ui, -apple-system, "Segoe UI", sans-serif; color: #0b0b0b;
         background: #f9f9f7; margin: 0; line-height: 1.6; }}
  .wrap {{ max-width: 1000px; margin: 0 auto; padding: 40px 24px 80px; }}
  header h1 {{ font-size: 28px; margin: 0 0 6px; }}
  header p {{ color: #52514e; margin: 0 0 8px; font-size: 15px; }}
  section {{ background: #fff; border: 1px solid rgba(11,11,11,.08); border-radius: 12px;
            padding: 24px 26px; margin: 26px 0; }}
  section h2 {{ font-size: 20px; margin: 0 0 12px; }}
  .analisis {{ font-size: 15px; color: #2a2a28; margin: 0 0 18px; }}
  .grafico {{ margin: 8px 0; }}
  footer {{ color: #898781; font-size: 13px; text-align: center; margin-top: 40px; }}
</style></head>
<body><div class="wrap">
<header>
  <h1>Uso de CREA: 2025 vs 2026</h1>
  <p>Análisis con drill-down — del panorama país al foco por contexto socioeconómico, grado y territorio.</p>
</header>
{secciones}
<footer>Generado con visualizaciones.py · datos anonimizados de Ceibal (abril–junio)</footer>
</div></body></html>"""


def main():
    print("Generando reporte único...")
    secciones = [seccion_panorama(), seccion_quintil(), seccion_quintil_grado(), seccion_mapa(),seccion_ciclo("estudiantes"),      # <-- NUEVO
                    seccion_ciclo("docentes"),]
    

    bloques, primero = [], True
    for titulo, texto, figuras in secciones:
        graficos = ""
        for fig in figuras:
            graficos += f'<div class="grafico">{fig.to_html(full_html=False, include_plotlyjs="cdn" if primero else False)}</div>'
            primero = False
        bloques.append(f'<section><h2>{titulo}</h2><p class="analisis">{texto}</p>{graficos}</section>')

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text(PLANTILLA.format(secciones="\n".join(bloques)), encoding="utf-8")
    print("  ->", SALIDA.resolve())


if __name__ == "__main__":
    main()
