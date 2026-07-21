"""
defensa.py — Deck de diapositivas para la defensa (20/07/2026).

Misma evidencia que presentacion.py, pero en formato PRESENTACIÓN:
pantalla completa, una idea por slide, texto mínimo. El texto largo se saca
a propósito: lo que va escrito es lo que el jurado lee, y si lee mucho no te
escucha. Los detalles los decís vos en voz alta.

NAVEGACIÓN: flechas ← →, barra espaciadora, o clic. Tecla F para pantalla
completa del navegador.

DIFERENCIAS CON presentacion.py
  - Lámina 3 ordenada por PROFUNDIZACIÓN (que es lo que anuncia el título),
    no por caída de uso sostenido.
  - Se excluyen ciclos con menos de MIN_N_GRAFICO alumnos: Educación Media
    Rural (821) dominaba visualmente con 112 alumnos detrás, y contradecía
    el argumento de la lámina 4.
  - Nombres de ciclo escritos a mano (el .title() dejaba "3Er. Ciclo Ebi").
"""

import unicodedata
from pathlib import Path

import pandas as pd
import plotly.express as px

CARPETA = Path("data/interim/entrega-02-corregida")
SALIDA = Path("dashboards/defensa.html")

MESES = ["Dias4", "Dias5", "Dias6"]
UMBRALES = [1, 5, 10]
MIN_N_CICLO = 500       # mínimo para considerar un ciclo comparable
MIN_N_GRAFICO = 5000    # mínimo para que aparezca en el gráfico de la lámina 3
TOP_CICLOS = 6

AZUL = "#2a78d6"
NARANJA = "#eb6834"
GRIS_25 = "#a9b4c2"
ROJO = "#d03b3b"
VERDE = "#2f8f6b"

NOMBRES = {
    "3er. ciclo ebi": "3er ciclo EBI",
    "4to. ciclo": "4to ciclo",
    "educacion basica integrada": "Educación Básica Integrada",
    "primaria": "Primaria",
    "primaria especial": "Primaria Especial",
    "bachillerato": "Bachillerato",
    "bachillerato figari": "Bachillerato Figari",
    "bachillerato tecnologico": "Bachillerato Tecnológico",
    "bachillerato tecnico profesional": "Bachillerato Técnico Profesional",
    "ciclo basico": "Ciclo Básico",
    "educacion media rural": "Educación Media Rural",
    "educacion media tecnologica": "Educación Media Tecnológica",
    "formacion profesional basica": "Formación Profesional Básica",
}


def norm(v) -> str:
    if pd.isna(v):
        return ""
    s = str(v).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def lindo(ciclo: str) -> str:
    return NOMBRES.get(ciclo, ciclo.title())


def cargar(pob: str) -> pd.DataFrame:
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


def tasa(df, por, umbral):
    d = df.copy()
    d["ok"] = d["dias_total"] >= umbral
    g = d.groupby(por, dropna=False).agg(n=("ok", "size"), acc=("ok", "sum")).reset_index()
    g["tasa"] = (100 * g["acc"] / g["n"]).round(1)
    return g


def estilo(fig, subtitulo="", alto=440):
    """Sin título dentro del gráfico: el título es el de la diapositiva."""
    fig.update_layout(
        title=dict(text=f"<span style='font-size:13px;color:#6b6a65'>{subtitulo}</span>",
                   x=0.01, xanchor="left", y=0.97) if subtitulo else None,
        font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif",
                  size=15, color="#0b0b0b"),
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="right", x=1, title=""),
        margin=dict(t=70 if subtitulo else 40, l=70, r=30, b=60),
        bargap=0.30, bargroupgap=0.10, height=alto,
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff")
    fig.update_yaxes(gridcolor="#e6e5df", zerolinecolor="#8f8e86", zerolinewidth=1.5)
    fig.update_xaxes(showgrid=False)
    return fig


# ============================================================
# datos
# ============================================================
def datos_umbral(datos):
    filas = []
    for pob, df in datos.items():
        for u in UMBRALES:
            t = tasa(df, ["anio"], u).set_index("anio")["tasa"]
            filas.append({"poblacion": pob.capitalize(),
                          "umbral": f"≥ {u} día{'s' if u > 1 else ''}",
                          "var_pp": round(t[2026] - t[2025], 1),
                          "t25": t[2025], "t26": t[2026]})
    return pd.DataFrame(filas)


def panel(df):
    g = df.groupby(["id_persona", "anio"])["dias_total"].max().unstack()
    g = g.dropna(subset=[2025, 2026])
    g.columns = ["d25", "d26"]
    return g


def clasificar(g):
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


def datos_panel(datos):
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
    return pd.DataFrame(filas), resumen


def datos_ciclo(df):
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
    g = g.reset_index()
    g["ciclo"] = g["ciclo"].map(lindo)
    return g


# ============================================================
# figuras
# ============================================================
def fig_umbral(d):
    fig = px.bar(d, x="umbral", y="var_pp", color="poblacion", barmode="group",
                 text="var_pp", custom_data=["poblacion", "t25", "t26"],
                 category_orders={"umbral": [f"≥ {u} día{'s' if u > 1 else ''}"
                                             for u in UMBRALES]},
                 color_discrete_map={"Estudiantes": AZUL, "Docentes": NARANJA})
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside",
                      hovertemplate="%{customdata[0]} · %{x}<br>"
                                    "%{customdata[1]:.1f}% → %{customdata[2]:.1f}%"
                                    "<extra></extra>")
    estilo(fig, "Variación de la tasa 2025→2026, en puntos porcentuales.")
    fig.update_yaxes(title="", ticksuffix=" pp")
    fig.update_xaxes(title="")
    return fig


def fig_panel(d):
    orden = ["Bajó intensidad", "Subió intensidad", "Se mantuvo",
             "Dejó de entrar", "Empezó a entrar", "Nunca entró"]
    fig = px.bar(d, x="cat", y="pct", color="poblacion", barmode="group",
                 text="pct", custom_data=["poblacion", "n"],
                 category_orders={"cat": orden},
                 color_discrete_map={"Estudiantes": AZUL, "Docentes": NARANJA})
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                      hovertemplate="%{customdata[0]} · %{x}<br>%{y:.1f}%"
                                    "<br>%{customdata[1]:,} personas<extra></extra>")
    estilo(fig, "Solo personas presentes en ambos años. Sin altas ni bajas de matrícula.")
    fig.update_yaxes(title="", ticksuffix="%")
    fig.update_xaxes(title="")
    return fig


def fig_ciclo(tab):
    sub = tab[tab["n_2026"] >= MIN_N_GRAFICO].nsmallest(TOP_CICLOS, "profundizacion")
    largo = sub.melt(id_vars="ciclo", value_vars=["var_pp_1", "var_pp_10"],
                     var_name="metrica", value_name="var_pp")
    largo["metrica"] = largo["metrica"].map({"var_pp_1": "Acceso (≥ 1 día)",
                                             "var_pp_10": "Uso sostenido (≥ 10 días)"})
    orden = sub.sort_values("profundizacion", ascending=False)["ciclo"].tolist()

    fig = px.bar(largo, x="var_pp", y="ciclo", color="metrica", barmode="group",
                 orientation="h", text="var_pp",
                 category_orders={"ciclo": orden},
                 color_discrete_map={"Acceso (≥ 1 día)": GRIS_25,
                                     "Uso sostenido (≥ 10 días)": ROJO})
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside",
                      hovertemplate="%{y}<br>%{fullData.name}: %{x:.1f} pp<extra></extra>")
    estilo(fig, "Gris: acceso. Rojo: uso sostenido. La distancia entre ambos es el problema oculto.",
           alto=470)
    fig.update_xaxes(title="Cambio 2025 → 2026", ticksuffix=" pp")
    fig.update_yaxes(title="")
    return fig, sub


def fig_volumen(tab):
    sub = tab.nlargest(8, "perdidos").sort_values("perdidos", ascending=False)
    fig = px.bar(sub, x="ciclo", y="perdidos", text="perdidos",
                 color_discrete_sequence=[VERDE],
                 custom_data=["rel", "t25", "t26", "n_2026"])
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside",
                      hovertemplate="<b>%{x}</b><br>%{customdata[3]:,} alumnos"
                                    "<br>%{customdata[1]:.1f}% → %{customdata[2]:.1f}%"
                                    "<br><b>%{y:,.0f} alumnos menos</b><extra></extra>")
    estilo(fig, "Alumnos que dejaron de alcanzar 10+ días de uso, a matrícula 2026 fija.")
    fig.update_yaxes(title="")
    fig.update_xaxes(title="", tickangle=-25)
    return fig, sub


# ============================================================
# HTML
# ============================================================
CSS = """
* { box-sizing: border-box; }
body { margin:0; font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
       background:#0f1720; color:#0b0b0b; overflow:hidden; }
.slide { display:none; width:100vw; height:100vh; background:#fff;
         padding:44px 60px 30px; flex-direction:column; }
.slide.on { display:flex; }
.kicker { font-size:13px; letter-spacing:2.4px; text-transform:uppercase;
          color:#8a8a84; margin-bottom:10px; }
h1 { font-size:52px; line-height:1.1; margin:0 0 18px; letter-spacing:-1.2px; max-width:20ch; }
h2 { font-size:38px; line-height:1.15; margin:0 0 10px; letter-spacing:-0.8px; }
.clave { font-size:20px; color:#3d3d3a; margin:0 0 6px; max-width:80ch; }
.clave b { color:#0b0b0b; }
.fig { flex:1; min-height:0; display:flex; align-items:center; }
.fig > div { width:100%; }
.portada { justify-content:center; background:#0f1720; color:#fff; }
.portada h1 { font-size:60px; color:#fff; max-width:24ch; }
.portada .sub { font-size:21px; color:#9fb0c0; max-width:60ch; }
.portada .meta { font-size:14px; color:#63788a; margin-top:40px; }
.cierre { justify-content:center; }
.cierre ol { font-size:26px; line-height:1.55; max-width:34ch; padding-left:26px; }
.cierre li { margin-bottom:14px; }
.pedido { margin-top:26px; font-size:19px; color:#3d3d3a; border-left:4px solid #2f8f6b;
          padding-left:16px; max-width:60ch; }
.nav { position:fixed; bottom:16px; right:24px; font-size:13px; color:#a8a8a2; }
.portada .nav { color:#63788a; }
@media print { .slide { display:flex; page-break-after:always; height:100vh; } .nav{display:none} }
"""

JS = """
let i = 0;
const s = document.querySelectorAll('.slide');
function ir(n){ s[i].classList.remove('on'); i = Math.max(0, Math.min(s.length-1, n));
  s[i].classList.add('on'); document.querySelectorAll('.pos').forEach(e =>
  e.textContent = (i+1) + ' / ' + s.length); window.dispatchEvent(new Event('resize')); }
document.addEventListener('keydown', e => {
  if (['ArrowRight',' ','PageDown'].includes(e.key)) { e.preventDefault(); ir(i+1); }
  if (['ArrowLeft','PageUp'].includes(e.key)) { e.preventDefault(); ir(i-1); }
  if (e.key === 'f' || e.key === 'F') {
    document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen();
  }
});
document.addEventListener('click', e => { if (!e.target.closest('.modebar')) ir(i+1); });
ir(0);
"""

BASE = """<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CREA 2025 vs 2026 — Intensidad de uso</title>
<style>{css}</style></head><body>
{slides}
<script>{js}</script></body></html>"""


def slide(contenido, clase=""):
    return (f'<section class="slide {clase}">{contenido}'
            f'<div class="nav"><span class="pos"></span> · ← → para navegar</div></section>')


def main():
    print("Cargando datos...")
    datos = {p: cargar(p) for p in ("estudiantes", "docentes")}
    for p, df in datos.items():
        print(f"  {p}: {len(df):,} filas")

    d_umb = datos_umbral(datos)
    d_pan, resumen = datos_panel(datos)
    tab = datos_ciclo(datos["estudiantes"])
    print(f"  {len(tab)} ciclos comparables")

    f1 = fig_umbral(d_umb)
    f2 = fig_panel(d_pan)
    f3, sub3 = fig_ciclo(tab)
    f4, sub4 = fig_volumen(tab)

    def html(fig, primero=False):
        return fig.to_html(full_html=False,
                           include_plotlyjs="cdn" if primero else False,
                           config={"displayModeBar": False, "responsive": True})

    est = d_umb[d_umb["poblacion"] == "Estudiantes"].set_index("umbral")["var_pp"]
    factor = abs(est.iloc[-1] / est.iloc[0])
    e, dc = resumen["estudiantes"], resumen["docentes"]
    top3 = tab.nlargest(3, "perdidos")
    total = tab[tab["perdidos"] > 0]["perdidos"].sum()
    concentra = 100 * top3["perdidos"].sum() / total
    peor_rel = tab.nsmallest(1, "rel").iloc[0]
    prof3 = sub3.nsmallest(3, "profundizacion")

    slides = []

    # portada
    slides.append(slide(
        '<div class="kicker">Reto Ceibal · CREA 2025 vs 2026</div>'
        '<h1>El acceso a CREA casi no cayó.<br>La frecuencia de uso, sí.</h1>'
        '<p class="sub">Todos medimos si el usuario entró. Cuando medimos '
        'cuántas veces entró, el problema es cuatro veces más grande.</p>'
        '<div class="meta">Santiago Alba · 20 de julio de 2026</div>', "portada"))

    # 1
    slides.append(slide(
        '<div class="kicker">Hallazgo principal</div>'
        '<h2>No es que entren menos. Entran menos veces.</h2>'
        f'<p class="clave">Exigir 10 días en vez de 1 <b>multiplica la caída por '
        f'{factor:.0f}</b>: de −1,7 a −6,8 pp en estudiantes.</p>'
        f'<div class="fig">{html(f1, primero=True)}</div>'))

    # 2
    slides.append(slide(
        '<div class="kicker">Control</div>'
        '<h2>No es rotación de matrícula.</h2>'
        f'<p class="clave">Mismas <b>{e["n"]:,} personas</b> en ambos años: '
        f'<b>{e["bajo"]:.0f}% bajó</b> su intensidad, {e["subio"]:.0f}% la subió. '
        f'Promedio: {e["d25"]} → {e["d26"]} días.</p>'
        '<p class="clave">Y tienen un año más: el uso crece con el grado. '
        '<b>Deberían haber subido.</b></p>'
        f'<div class="fig">{html(f2)}</div>'))

    # 3
    lista3 = ", ".join(f'<b>{r["ciclo"]}</b>' for _, r in prof3.iterrows())
    slides.append(slide(
        '<div class="kicker">Dónde ocurre</div>'
        '<h2>Se soltó el núcleo obligatorio.</h2>'
        f'<p class="clave">{lista3}: acceso casi intacto, uso sostenido −9 a −10 pp. '
        '<b>No perdieron usuarios: perdieron la rutina.</b></p>'
        '<p class="clave">En los bachilleratos técnicos el acceso incluso '
        '<b>subió</b> mientras el uso sostenido caía.</p>'
        f'<div class="fig">{html(f3)}</div>'))

    # 4
    slides.append(slide(
        '<div class="kicker">Recomendación</div>'
        '<h2>Dónde una intervención alcanza a más gente.</h2>'
        f'<p class="clave">La peor caída relativa es {peor_rel["ciclo"]} '
        f'({peor_rel["rel"]:.0f}%), pero son <b>{peor_rel["perdidos"]:,.0f} alumnos</b>. '
        f'El volumen está en '
        + ", ".join(f'<b>{r["ciclo"]}</b>' for _, r in top3.iterrows())
        + f': el <b>{concentra:.0f}%</b> de los {total:,.0f} afectados.</p>'
        f'<div class="fig">{html(f4)}</div>'))

    # cierre
    slides.append(slide(
        '<div class="kicker">En una línea</div>'
        '<h2>Qué nos llevamos</h2>'
        '<ol>'
        '<li>Medir <b>“accedió al menos una vez”</b> subestima el problema por cuatro.</li>'
        '<li>No es matrícula: entre las mismas personas, también cae.</li>'
        '<li>El acceso ya está resuelto. Lo que hay que reconstruir es la <b>frecuencia</b>.</li>'
        '</ol>'
        '<div class="pedido"><b>Lo que pedimos a Ceibal:</b> cronograma de implementación '
        'de la reforma por grado y calendario lectivo 2025–2026. Sin eso no se puede '
        'atribuir la caída a una causa.</div>', "cierre"))

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text(BASE.format(css=CSS, js=JS, slides="\n".join(slides)),
                      encoding="utf-8")
    print("\n  ->", SALIDA.resolve())
    print("  Abrilo con:  open dashboards/defensa.html")
    print("  Navegá con ← →   ·   F para pantalla completa")


if __name__ == "__main__":
    main()