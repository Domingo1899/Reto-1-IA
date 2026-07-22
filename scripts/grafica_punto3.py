from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# CSV generados por scripts/docente_estudiante.py
CSV_CUARTILES = Path("resultado_docente_estudiante_cuartiles.csv")
CSV_RESUMEN = Path("resultado_docente_estudiante_resumen.csv")

SALIDA_HTML = Path("03_asociacion_docente_estudiante.html")
SALIDA_PNG = Path("03_asociacion_docente_estudiante.png")
SALIDA_SVG = Path("03_asociacion_docente_estudiante.svg")

PERIODO = "Mayo-Junio"
ORDEN_CUARTILES = [
    "Q1 mayor caída docente",
    "Q2",
    "Q3",
    "Q4 mejor evolución docente",
]

COLORES = {
    "Secundaria": "#7C3AED",
    "UTU": "#F05A0A",
}


def numero_es(valor: float, decimales: int = 2) -> str:
    """Formatea un número con coma decimal."""
    return f"{valor:.{decimales}f}".replace(".", ",")


def cargar_datos() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not CSV_CUARTILES.exists():
        raise FileNotFoundError(
            f"No se encontró {CSV_CUARTILES}. "
            "Ejecutá primero scripts/docente_estudiante.py."
        )

    if not CSV_RESUMEN.exists():
        raise FileNotFoundError(
            f"No se encontró {CSV_RESUMEN}. "
            "Ejecutá primero scripts/docente_estudiante.py."
        )

    cuartiles = pd.read_csv(CSV_CUARTILES, sep=";", decimal=",")
    resumen = pd.read_csv(CSV_RESUMEN, sep=";", decimal=",")

    columnas_cuartiles = {
        "subsistema",
        "periodo",
        "cuartil_docente",
        "centros",
        "delta_docente_promedio",
        "delta_estudiante_promedio",
    }
    faltantes = columnas_cuartiles - set(cuartiles.columns)
    if faltantes:
        raise ValueError(
            "Faltan columnas en resultado_docente_estudiante_cuartiles.csv: "
            f"{sorted(faltantes)}"
        )

    columnas_resumen = {
        "subsistema",
        "periodo",
        "pearson_delta_promedio",
        "spearman_delta_promedio",
    }
    faltantes = columnas_resumen - set(resumen.columns)
    if faltantes:
        raise ValueError(
            "Faltan columnas en resultado_docente_estudiante_resumen.csv: "
            f"{sorted(faltantes)}"
        )

    cuartiles = cuartiles[
        (cuartiles["periodo"] == PERIODO)
        & cuartiles["subsistema"].isin(["Secundaria", "UTU"])
    ].copy()

    resumen = resumen[
        (resumen["periodo"] == PERIODO)
        & resumen["subsistema"].isin(["Secundaria", "UTU"])
    ].copy()

    if cuartiles.empty:
        raise ValueError(
            f"No hay filas para periodo={PERIODO!r} en {CSV_CUARTILES}."
        )

    return cuartiles, resumen


def preparar_subsistema(
    cuartiles: pd.DataFrame,
    subsistema: str,
) -> pd.DataFrame:
    datos = cuartiles[cuartiles["subsistema"] == subsistema].copy()

    datos["cuartil_docente"] = pd.Categorical(
        datos["cuartil_docente"],
        categories=ORDEN_CUARTILES,
        ordered=True,
    )
    datos = datos.sort_values("cuartil_docente")

    if len(datos) != 4:
        raise ValueError(
            f"Se esperaban 4 cuartiles para {subsistema}, pero hay {len(datos)}."
        )

    etiquetas = []
    for _, fila in datos.iterrows():
        nombre = str(fila["cuartil_docente"])
        if nombre.startswith("Q1"):
            titulo = "Q1<br>Mayor caída docente"
        elif nombre.startswith("Q4"):
            titulo = "Q4<br>Mejor evolución docente"
        else:
            titulo = nombre

        delta_docente = numero_es(fila["delta_docente_promedio"], 2)
        etiquetas.append(
            f"{titulo}<br><span style='font-size:12px'>"
            f"Docentes: {delta_docente} días</span>"
        )

    datos["etiqueta_x"] = etiquetas
    datos["etiqueta_barra"] = datos["delta_estudiante_promedio"].map(
        lambda x: f"{numero_es(x, 2)} días"
    )

    return datos


def obtener_correlacion(
    resumen: pd.DataFrame,
    subsistema: str,
) -> tuple[float, float]:
    fila = resumen[resumen["subsistema"] == subsistema]
    if fila.empty:
        return float("nan"), float("nan")

    return (
        float(fila.iloc[0]["pearson_delta_promedio"]),
        float(fila.iloc[0]["spearman_delta_promedio"]),
    )


def construir_figura(
    cuartiles: pd.DataFrame,
    resumen: pd.DataFrame,
) -> go.Figure:
    sec = preparar_subsistema(cuartiles, "Secundaria")
    utu = preparar_subsistema(cuartiles, "UTU")

    r_sec, rho_sec = obtener_correlacion(resumen, "Secundaria")
    r_utu, rho_utu = obtener_correlacion(resumen, "UTU")

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.12,
        subplot_titles=(
            f"Secundaria · Pearson r={numero_es(r_sec, 2)}",
            f"UTU · Pearson r={numero_es(r_utu, 2)}",
        ),
    )

    for columna, subsistema, datos in [
        (1, "Secundaria", sec),
        (2, "UTU", utu),
    ]:
        fig.add_trace(
            go.Bar(
                x=datos["etiqueta_x"],
                y=datos["delta_estudiante_promedio"],
                marker_color=COLORES[subsistema],
                text=datos["etiqueta_barra"],
                textposition="outside",
                customdata=datos[
                    [
                        "centros",
                        "delta_docente_promedio",
                        "delta_estudiante_promedio",
                    ]
                ],
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Centros: %{customdata[0]:.0f}<br>"
                    "Cambio docente: %{customdata[1]:.2f} días<br>"
                    "Cambio estudiantil: %{customdata[2]:.2f} días"
                    "<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=columna,
        )

        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="#64748B",
            line_width=1.5,
            row=1,
            col=columna,
        )

    # Diferencia entre el peor y el mejor cuartil.
    diferencia_sec = abs(
        sec.iloc[0]["delta_estudiante_promedio"]
        - sec.iloc[-1]["delta_estudiante_promedio"]
    )
    diferencia_utu = abs(
        utu.iloc[0]["delta_estudiante_promedio"]
        - utu.iloc[-1]["delta_estudiante_promedio"]
    )

    fig.add_annotation(
        x=0.22,
        y=1.04,
        xref="paper",
        yref="paper",
        text=(
            f"Brecha Q1–Q4: <b>{numero_es(diferencia_sec, 2)} días</b>"
            f"<br><span style='font-size:12px'>Spearman ρ={numero_es(rho_sec, 2)}</span>"
        ),
        showarrow=False,
        align="center",
        font=dict(size=14, color="#334155"),
    )

    fig.add_annotation(
        x=0.78,
        y=1.04,
        xref="paper",
        yref="paper",
        text=(
            f"Brecha Q1–Q4: <b>{numero_es(diferencia_utu, 2)} días</b>"
            f"<br><span style='font-size:12px'>Spearman ρ={numero_es(rho_utu, 2)}</span>"
        ),
        showarrow=False,
        align="center",
        font=dict(size=14, color="#334155"),
    )

    fig.update_yaxes(
        title_text="Cambio promedio de días de uso estudiantil",
        ticksuffix=" días",
        range=[-7, 0.8],
        gridcolor="#E2E8F0",
        zeroline=False,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="",
        ticksuffix=" días",
        range=[-7, 0.8],
        gridcolor="#E2E8F0",
        zeroline=False,
        row=1,
        col=2,
    )

    fig.update_xaxes(
        title_text="Centros agrupados según cambio docente",
        tickfont=dict(size=13),
        row=1,
        col=1,
    )
    fig.update_xaxes(
        title_text="Centros agrupados según cambio docente",
        tickfont=dict(size=13),
        row=1,
        col=2,
    )

    fig.update_layout(
        title={
            "text": (
                "<b>3. Cuanto más cae el uso docente, más cae el uso estudiantil</b>"
                "<br><span style='font-size:17px'>"
                "Cambio acumulado en mayo-junio, comparando 2026 con 2025 por centro"
                "</span>"
            ),
            "x": 0.03,
            "xanchor": "left",
            "y": 0.98,
            "yanchor": "top",
        },
        template="plotly_white",
        width=1600,
        height=900,
        margin=dict(t=150, l=95, r=55, b=150),
        font=dict(family="Arial", size=15, color="#17233B"),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    fig.add_annotation(
        x=0.5,
        y=-0.21,
        xref="paper",
        yref="paper",
        text=(
            "Lectura: Q1 reúne los centros con mayor caída docente y Q4 los de mejor evolución. "
            "Las barras muestran la variación promedio de días de uso de los estudiantes. "
            "La asociación es institucional y no demuestra causalidad."
        ),
        showarrow=False,
        align="center",
        font=dict(size=14, color="#475569"),
    )

    return fig


def main() -> None:
    cuartiles, resumen = cargar_datos()
    fig = construir_figura(cuartiles, resumen)

    fig.write_html(SALIDA_HTML, include_plotlyjs="cdn")

    try:
        fig.write_image(SALIDA_PNG, scale=2)
        fig.write_image(SALIDA_SVG)
    except Exception as error:
        print(
            "No se pudieron exportar PNG/SVG. "
            "Instalá o actualizá kaleido con: pip install -U kaleido"
        )
        print(f"Detalle: {error}")

    print("\nArchivos generados:")
    print(SALIDA_HTML)
    print(SALIDA_PNG)
    print(SALIDA_SVG)


if __name__ == "__main__":
    main()