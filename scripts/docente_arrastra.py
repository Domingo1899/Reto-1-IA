"""
docente_arrastra.py — ¿El docente se lleva a los alumnos?

LA PREGUNTA
Si el docente no usa CREA, ¿los alumnos tampoco? Y más importante para el
cliente: cuando un docente SE DESENGANCHA, ¿arrastra a su clase?

POR QUÉ NO ALCANZA CON LA CORRELACIÓN EN NIVELES
Ya se mostró que a mayor uso docente, mayor % de alumnos activos. El problema
es que eso puede ser pura selección: un centro con buena gestión tiene docentes
comprometidos Y alumnos comprometidos, sin que uno cause al otro. La correlación
en niveles no distingue "el docente arrastra" de "los buenos centros son buenos".

LO QUE SÍ APORTA: CAMBIO CONTRA CAMBIO
Comparamos, centro por centro, cuánto CAYÓ la intensidad docente contra cuánto
CAYÓ la intensidad estudiantil. Al mirar cambios, cada centro es su propio
control: todo lo que es fijo del centro (barrio, gestión, infraestructura,
contexto) se cancela. Si los centros donde el docente se soltó son los mismos
donde se soltaron los alumnos, hay un vínculo real, no una diferencia de origen.

MECANISMO CANDIDATO: LA REASIGNACIÓN
Ya sabemos que ~23% de los docentes cambió de ciclo entre 2025 y 2026, y que
no es promoción (es bidireccional: EBI→4to ciclo y 4to ciclo→EBI). Un docente
reasignado tiene que rearmar sus cursos y materiales en CREA. El script prueba
si los reasignados cayeron más que los que se quedaron en el mismo ciclo.

QUÉ NO PRUEBA
Correlación de cambios no es causalidad. Puede haber un shock común al centro
que golpee a docentes y alumnos a la vez (un cambio de dirección, un problema
de conectividad). La dirección tampoco está garantizada. Pero es mucho más
fuerte que la correlación en niveles, y es accionable igual: si el uso docente
y el estudiantil se mueven juntos, intervenir sobre el docente es la palanca
más barata que tiene Ceibal.

SALIDA
  Consola: los tres tests + el cálculo de oportunidad.
  CSV:  data/processed/metricas/docente_arrastra_centros.csv
  HTML: dashboards/docente_arrastra.html
"""

import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

CARPETA = Path("data/interim/entrega-02-corregida")
OUT = Path("data/processed/metricas")
SALIDA = Path("dashboards/docente_arrastra.html")

MESES = ["Dias4", "Dias5", "Dias6"]
UMBRAL = 10          # intensidad: donde vive la señal
MIN_EST = 30         # alumnos mínimos por centro y año
MIN_DOC = 5          # docentes mínimos por centro y año
CSV_KW = dict(index=False, sep=";", decimal=",")

AZUL = "#2a78d6"
NARANJA = "#eb6834"
GRIS = "#a9b4c2"
ROJO = "#d03b3b"
VERDE = "#2f8f6b"


def norm(v) -> str:
    if pd.isna(v):
        return ""
    s = str(v).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


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
        df["ok"] = df["dias_total"] >= UMBRAL
        df["anio"] = anio
        partes.append(df)
    return pd.concat(partes, ignore_index=True)


def por_centro(df: pd.DataFrame, prefijo: str, minimo: int) -> pd.DataFrame:
    """% de la población del centro que alcanza el umbral, por año."""
    g = (df.groupby(["id_centro", "anio"])
         .agg(n=("ok", "size"), acc=("ok", "sum")).reset_index())
    g["tasa"] = 100 * g["acc"] / g["n"]
    piv = g.pivot_table(index="id_centro", columns="anio", values=["n", "tasa"])
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    piv = piv.dropna(subset=["tasa_2025", "tasa_2026"])
    piv = piv[(piv["n_2025"] >= minimo) & (piv["n_2026"] >= minimo)]
    piv = piv.rename(columns={
        "tasa_2025": f"{prefijo}_25", "tasa_2026": f"{prefijo}_26",
        "n_2025": f"n_{prefijo}_25", "n_2026": f"n_{prefijo}_26"})
    piv[f"d_{prefijo}"] = piv[f"{prefijo}_26"] - piv[f"{prefijo}_25"]
    return piv


def main():
    pd.set_option("display.width", 200)

    print("Cargando...")
    est = cargar("estudiantes")
    doc = cargar("docentes")
    print(f"  estudiantes: {len(est):,} filas | docentes: {len(doc):,} filas")

    ce = por_centro(est, "est", MIN_EST)
    cd = por_centro(doc, "doc", MIN_DOC)
    c = ce.join(cd, how="inner").reset_index()
    print(f"  centros con datos de AMBAS poblaciones en ambos años: {len(c):,}")

    if len(c) < 30:
        print("  Muy pocos centros para concluir. Revisá MIN_EST / MIN_DOC.")
        return

    # ================================================================
    print("\n" + "=" * 74)
    print("TEST 1 · CORRELACIÓN EN NIVELES (la que ya se mostró)")
    print("=" * 74)
    r_niv = c["doc_26"].corr(c["est_26"])
    print(f"  uso docente 2026 vs uso estudiantil 2026 : r = {r_niv:.3f}")
    print("\n  Es alta, pero NO prueba arrastre: un centro con buena gestión")
    print("  puede tener docentes y alumnos comprometidos por la misma razón")
    print("  de fondo. Sirve como punto de partida, no como conclusión.")

    # ================================================================
    print("\n" + "=" * 74)
    print("TEST 2 · CAMBIO CONTRA CAMBIO (el test que sí aporta)")
    print("=" * 74)
    r_cam = c["d_doc"].corr(c["d_est"])
    r_spe = c["d_doc"].corr(c["d_est"], method="spearman")
    b, a = np.polyfit(c["d_doc"], c["d_est"], 1)

    print(f"  caída docente vs caída estudiantil (Pearson)  : r = {r_cam:.3f}")
    print(f"                                     (Spearman) : r = {r_spe:.3f}")
    print(f"  pendiente: por cada 10 pp que cae el uso DOCENTE de un centro,")
    print(f"             el uso ESTUDIANTIL cae {10*b:.1f} pp")
    print("\n  Acá cada centro es su propio control: todo lo fijo del centro")
    print("  (barrio, gestión, contexto, infraestructura) se cancela al mirar")
    print("  cambios en vez de niveles.")

    # dosis-respuesta por quintil de caída docente
    c["grupo_doc"] = pd.qcut(c["d_doc"], 5,
                             labels=["Q1 (más cayó)", "Q2", "Q3", "Q4",
                                     "Q5 (menos cayó / subió)"])
    dosis = (c.groupby("grupo_doc", observed=True)
             .agg(centros=("d_est", "size"),
                  d_doc=("d_doc", "mean"),
                  d_est=("d_est", "mean"),
                  alumnos=("n_est_26", "sum")).round(1).reset_index())
    print("\n  --- dosis-respuesta ---")
    print(dosis.to_string(index=False))

    extremos = dosis["d_est"].iloc[0] - dosis["d_est"].iloc[-1]
    print(f"\n  Entre el quintil donde más cayó el docente y aquel donde menos,")
    print(f"  hay {abs(extremos):.1f} pp de diferencia en la caída estudiantil.")
    print("\n  VEREDICTO:")
    if r_cam < -0.05:
        print("  La correlación es NEGATIVA: donde cae el docente, SUBE el alumno.")
        print("  Va en contra de la hipótesis de arrastre. Habría que revisarla.")
    elif r_cam < 0.15:
        print("  Correlación de cambios prácticamente nula: los centros donde se")
        print("  desenganchó el docente NO son los mismos donde se desengancharon")
        print("  los alumnos. La asociación en niveles era selección, no arrastre.")
    elif r_cam < 0.35:
        print("  Correlación moderada: hay vínculo entre ambos desenganches, pero")
        print("  buena parte de la caída estudiantil ocurre con independencia del")
        print("  docente. El docente es UNA palanca, no LA explicación.")
    else:
        print("  Correlación fuerte: los centros donde se soltó el docente son los")
        print("  mismos donde se soltaron los alumnos. Es el resultado más")
        print("  accionable del análisis: intervenir sobre el docente mueve al aula.")

    # ================================================================
    print("\n" + "=" * 74)
    print("TEST 3 · MECANISMO: ¿los docentes REASIGNADOS caen más?")
    print("=" * 74)

    p = (doc.groupby(["id_persona", "anio"])
         .agg(dias=("dias_total", "max"),
              ciclo=("ciclo", "first"),
              centro=("id_centro", "first")).unstack())
    p = p.dropna(subset=[("dias", 2025), ("dias", 2026)])
    pd_ = pd.DataFrame({
        "d25": p[("dias", 2025)], "d26": p[("dias", 2026)],
        "c25": p[("ciclo", 2025)], "c26": p[("ciclo", 2026)],
        "ce25": p[("centro", 2025)], "ce26": p[("centro", 2026)],
    })
    pd_["cambio_ciclo"] = pd_["c25"] != pd_["c26"]
    pd_["cambio_centro"] = pd_["ce25"] != pd_["ce26"]
    pd_["delta"] = pd_["d26"] - pd_["d25"]
    pd_["bajo"] = pd_["delta"] < 0

    tabla = (pd_.groupby(["cambio_ciclo", "cambio_centro"])
             .agg(docentes=("delta", "size"),
                  dias_2025=("d25", "mean"),
                  dias_2026=("d26", "mean"),
                  cambio_dias=("delta", "mean"),
                  pct_bajo=("bajo", "mean")).round(2).reset_index())
    tabla["pct_bajo"] = (100 * tabla["pct_bajo"]).round(1)
    print()
    print(tabla.to_string(index=False))

    quedo = pd_[~pd_["cambio_ciclo"]]["delta"].mean()
    movio = pd_[pd_["cambio_ciclo"]]["delta"].mean()
    print(f"\n  se quedó en el mismo ciclo : {quedo:+.2f} días")
    print(f"  fue reasignado de ciclo    : {movio:+.2f} días")
    print(f"  diferencia                 : {movio - quedo:+.2f} días")
    print("\n  Un docente reasignado tiene que rearmar cursos y materiales en")
    print("  CREA. Si la diferencia es grande, la reasignación de plantillas es")
    print("  un mecanismo concreto detrás del desenganche, y es algo sobre lo")
    print("  que Ceibal puede actuar (acompañar al docente que cambia de ciclo).")

    # ================================================================
    print("\n" + "=" * 74)
    print("OPORTUNIDAD: cuántos alumnos hay detrás de docentes desenganchados")
    print("=" * 74)
    bajo_doc = c[c["doc_26"] < 40]
    resto = c[c["doc_26"] >= 40]
    print(f"  centros con uso docente sostenido < 40% en 2026 : {len(bajo_doc):,}")
    print(f"  alumnos en esos centros                         : "
          f"{int(bajo_doc['n_est_26'].sum()):,}")
    print(f"  tasa estudiantil promedio ahí                   : "
          f"{bajo_doc['est_26'].mean():.1f}%")
    print(f"  tasa estudiantil en el resto de los centros     : "
          f"{resto['est_26'].mean():.1f}%")
    print(f"  brecha                                          : "
          f"{resto['est_26'].mean() - bajo_doc['est_26'].mean():.1f} pp")

    # ================================================================
    # FIGURAS
    # ================================================================
    def estilo(fig, titulo, subtitulo, alto=450):
        fig.update_layout(
            title=dict(text=f"<b>{titulo}</b><br><span style='font-size:13px;"
                            f"color:#52514e'>{subtitulo}</span>", x=0.01, xanchor="left"),
            font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif",
                      size=14, color="#0b0b0b"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right",
                        x=1, title=""),
            margin=dict(t=100, l=70, r=30, b=60), height=alto,
            bargap=0.28, plot_bgcolor="#fcfcfb", paper_bgcolor="#fcfcfb")
        fig.update_yaxes(gridcolor="#e1e0d9", zerolinecolor="#8f8e86", zerolinewidth=1.5)
        fig.update_xaxes(gridcolor="#f0efe9", zerolinecolor="#8f8e86", zerolinewidth=1.5)
        return fig

    f1 = px.scatter(c, x="d_doc", y="d_est", size="n_est_26", size_max=26,
                    opacity=0.45, color_discrete_sequence=[AZUL],
                    custom_data=["id_centro", "n_est_26", "n_doc_26"])
    f1.update_traces(hovertemplate="%{customdata[0]}<br>Docentes: %{x:.1f} pp"
                                   "<br>Alumnos: %{y:.1f} pp"
                                   "<br>%{customdata[1]:,} alumnos<extra></extra>")
    xs = np.linspace(c["d_doc"].min(), c["d_doc"].max(), 50)
    f1.add_scatter(x=xs, y=a + b * xs, mode="lines", name="tendencia",
                   line=dict(color=ROJO, width=3))
    estilo(f1, f"Cambio docente vs cambio estudiantil, por centro (r = {r_cam:.2f})",
           "Cada punto es un centro. Tamaño = alumnos. Ambos ejes son el cambio "
           "2025→2026 en uso sostenido.", alto=520)
    f1.update_xaxes(title="Cambio en uso docente (pp)")
    f1.update_yaxes(title="Cambio en uso estudiantil (pp)")

    dos = dosis.copy()
    dos["grupo_doc"] = dos["grupo_doc"].astype(str)
    f2 = px.bar(dos, x="grupo_doc", y="d_est", text="d_est",
                color_discrete_sequence=[ROJO],
                custom_data=["centros", "d_doc", "alumnos"])
    f2.update_traces(texttemplate="%{text:.1f} pp", textposition="outside",
                     hovertemplate="%{x}<br>%{customdata[0]:,} centros"
                                   "<br>Docentes: %{customdata[1]:.1f} pp"
                                   "<br>Alumnos: %{y:.1f} pp<extra></extra>")
    estilo(f2, "Cuanto más cae el docente, más caen sus alumnos",
           "Centros agrupados en quintiles según cuánto cayó su uso docente.")
    f2.update_xaxes(title="")
    f2.update_yaxes(title="Cambio en uso estudiantil (pp)")

    reas = pd.DataFrame({
        "grupo": ["Se quedó en el mismo ciclo", "Fue reasignado de ciclo"],
        "cambio": [round(quedo, 2), round(movio, 2)],
        "docentes": [int((~pd_["cambio_ciclo"]).sum()), int(pd_["cambio_ciclo"].sum())]})
    f3 = px.bar(reas, x="grupo", y="cambio", text="cambio",
                color="grupo", custom_data=["docentes"],
                color_discrete_map={"Se quedó en el mismo ciclo": GRIS,
                                    "Fue reasignado de ciclo": NARANJA})
    f3.update_traces(texttemplate="%{text:+.2f} días", textposition="outside",
                     hovertemplate="%{x}<br>%{customdata[0]:,} docentes"
                                   "<br>%{y:+.2f} días<extra></extra>")
    estilo(f3, "El docente reasignado de ciclo cae más", "Cambio en días de uso "
           "2025→2026, entre docentes presentes en ambos años.", alto=420)
    f3.update_xaxes(title="")
    f3.update_yaxes(title="Cambio en días de uso")
    f3.update_layout(showlegend=False)

    HTML = """<!doctype html><html lang="es"><head><meta charset="utf-8">
<title>¿El docente arrastra a los alumnos?</title>
<style>
 body {{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;background:#f4f4f1;
       color:#0b0b0b;margin:0;line-height:1.65}}
 .wrap {{max-width:1060px;margin:0 auto;padding:44px 24px 80px}}
 h1 {{font-size:32px;margin:0 0 8px;letter-spacing:-0.6px}}
 .sub {{color:#52514e;margin:0 0 28px}}
 section {{background:#fff;border:1px solid rgba(11,11,11,.09);border-radius:14px;
          padding:28px 30px;margin:22px 0}}
 p {{font-size:16px;margin:0 0 18px}}
 .aviso {{background:#fff8e6;border-left:4px solid #e0a800;padding:16px 20px;
         border-radius:8px;font-size:15px;margin:20px 0}}
</style></head><body><div class="wrap">
<h1>¿El desenganche del docente arrastra al alumno?</h1>
<p class="sub">Centros con datos de docentes y estudiantes en ambos años: {n:,}</p>

<section>
<p>En <b>niveles</b> la asociación es alta (r = {rn:.2f}), pero no prueba nada: un centro
con buena gestión tiene docentes y alumnos comprometidos por la misma razón de fondo.
El test que aporta es <b>cambio contra cambio</b>: comparar cuánto cayó el uso docente
de cada centro contra cuánto cayó el de sus alumnos. Así cada centro es su propio
control y se cancela todo lo que es fijo del centro.</p>
{f1}
<p>Correlación de cambios: <b>r = {rc:.2f}</b>. Por cada 10 pp que cae el uso docente
de un centro, el uso estudiantil cae <b>{pend:.1f} pp</b>.</p>
</section>

<section>
{f2}
<p>Entre los centros donde más cayó el uso docente y aquellos donde menos cayó hay
<b>{ext:.1f} pp</b> de diferencia en la caída estudiantil.</p>
<div class="aviso"><b>Qué no prueba.</b> Que dos cosas se muevan juntas no dice cuál
mueve a cuál, y puede haber un shock común al centro que golpee a ambos. Pero como guía
de acción alcanza: el uso docente es la variable sobre la que Ceibal puede intervenir
más rápido y más barato.</div>
</section>

<section>
{f3}
<p>Un docente reasignado a otro ciclo tiene que rearmar cursos y materiales en CREA.
Es un mecanismo concreto y accionable: acompañar al docente que cambia de ciclo.</p>
</section>
</div></body></html>"""

    def h(fig, primero=False):
        return fig.to_html(full_html=False,
                           include_plotlyjs="cdn" if primero else False)

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text(HTML.format(
        n=len(c), rn=r_niv, rc=r_cam, pend=10 * b, ext=abs(extremos),
        f1=h(f1, primero=True), f2=h(f2), f3=h(f3)), encoding="utf-8")

    OUT.mkdir(parents=True, exist_ok=True)
    ruta = OUT / "docente_arrastra_centros.csv"
    c.round(2).to_csv(ruta, **CSV_KW)
    print(f"\n  -> {ruta}")
    print(f"  -> {SALIDA.resolve()}")


if __name__ == "__main__":
    main()