from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


ENTRADA = Path("resultado_focos_secundaria_utu.csv")
SALIDA_HTML = Path("grafica_transiciones_clave.html")
SALIDA_PNG = Path("grafica_transiciones_clave.png")
SALIDA_SVG = Path("grafica_transiciones_clave.svg")


def cargar_datos() -> pd.DataFrame:
    df = pd.read_csv(ENTRADA, sep=";", decimal=",")

    # Nos quedamos con la tabla de transiciones de grado
    df = df[df["dimension"] == "transicion_grado"].copy()

    # Solo Secundaria y UTU
    df = df[df["subsistema"].isin(["Secundaria", "UTU"])].copy()

    # Normalizamos algunos nombres para que queden prolijos
    df["grupo"] = (
        df["grupo"]
        .astype(str)
        .str.replace("→", "→", regex=False)
        .str.replace("  ", " ", regex=False)
        .str.strip()
    )

    return df


def preparar_top(df: pd.DataFrame, subsistema: str, top_n: int = 5) -> pd.DataFrame:
    sub = df[df["subsistema"] == subsistema].copy()

    # Ordenamos por contribución a la pérdida
    sub = sub.sort_values("contribucion_perdida_pct", ascending=False).head(top_n).copy()

    # Para que en la gráfica horizontal quede la mayor arriba
    sub = sub.sort_values("contribucion_perdida_pct", ascending=True).copy()

    sub["etiqueta_barra"] = (
        sub["grupo"]
        + "<br>"
        + "Estudiantes: "
        + sub["personas"].map(lambda x: f"{int(x):,}".replace(",", "."))
        + " | "
        + "Caída: "
        + sub["diferencia_promedio"].map(lambda x: f"{x:.2f}".replace(".", ","))
        + " días"
    )

    sub["etiqueta_texto"] = sub["contribucion_perdida_pct"].map(
        lambda x: f"{x:.1f}%".replace(".", ",")
    )

    return sub


def construir_figura(sec: pd.DataFrame, utu: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Secundaria: transiciones que más aportan a la caída",
            "UTU: transiciones que más aportan a la caída",
        ),
        horizontal_spacing=0.18,
    )

    color_sec = "#6C4CF1"
    color_utu = "#E16A2B"

    fig.add_trace(
        go.Bar(
            x=sec["contribucion_perdida_pct"],
            y=sec["grupo"],
            orientation="h",
            text=sec["etiqueta_texto"],
            textposition="outside",
            marker_color=color_sec,
            customdata=sec[["personas", "diferencia_promedio", "contribucion_perdida_pct"]],
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Estudiantes: %{customdata[0]:,.0f}<br>"
                "Caída promedio: %{customdata[1]:.2f} días<br>"
                "Aporte a la pérdida total: %{customdata[2]:.2f}%<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=utu["contribucion_perdida_pct"],
            y=utu["grupo"],
            orientation="h",
            text=utu["etiqueta_texto"],
            textposition="outside",
            marker_color=color_utu,
            customdata=utu[["personas", "diferencia_promedio", "contribucion_perdida_pct"]],
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Estudiantes: %{customdata[0]:,.0f}<br>"
                "Caída promedio: %{customdata[1]:.2f} días<br>"
                "Aporte a la pérdida total: %{customdata[2]:.2f}%<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title={
            "text": (
                "<b>La caída se concentra en transiciones educativas específicas</b>"
                "<br><span style='font-size:16px;'>"
                "Cada barra muestra qué porcentaje de la pérdida total de uso proviene de esa transición."
                "</span>"
            ),
            "x": 0.5,
            "xanchor": "center",
        },
        template="plotly_white",
        width=1500,
        height=800,
        margin=dict(t=120, l=70, r=70, b=80),
        font=dict(size=16),
    )

    fig.update_xaxes(
        title_text="Porcentaje de la pérdida total",
        ticksuffix="%",
        showgrid=True,
        zeroline=False,
        row=1,
        col=1,
    )
    fig.update_xaxes(
        title_text="Porcentaje de la pérdida total",
        ticksuffix="%",
        showgrid=True,
        zeroline=False,
        row=1,
        col=2,
    )

    fig.update_yaxes(title_text="", row=1, col=1)
    fig.update_yaxes(title_text="", row=1, col=2)

    # Agregamos una nota abajo
    fig.add_annotation(
        text=(
            "Lectura sugerida: la barra indica cuánto aporta cada transición a la pérdida total; "
            "el detalle de estudiantes y caída promedio se ve al pasar el mouse."
        ),
        xref="paper",
        yref="paper",
        x=0.5,
        y=-0.12,
        showarrow=False,
        font=dict(size=14, color="gray"),
    )

    return fig


def main():
    df = cargar_datos()

    sec = preparar_top(df, "Secundaria", top_n=5)
    utu = preparar_top(df, "UTU", top_n=5)

    print("\nTOP TRANSICIONES - SECUNDARIA")
    print(
        sec[
            [
                "grupo",
                "personas",
                "diferencia_promedio",
                "variacion_promedio_pct",
                "contribucion_perdida_pct",
            ]
        ].to_string(index=False)
    )

    print("\nTOP TRANSICIONES - UTU")
    print(
        utu[
            [
                "grupo",
                "personas",
                "diferencia_promedio",
                "variacion_promedio_pct",
                "contribucion_perdida_pct",
            ]
        ].to_string(index=False)
    )

    fig = construir_figura(sec, utu)

    fig.write_html(SALIDA_HTML)
    fig.write_image(SALIDA_PNG, scale=2)
    fig.write_image(SALIDA_SVG)

    print("\nArchivos generados:")
    print(SALIDA_HTML)
    print(SALIDA_PNG)
    print(SALIDA_SVG)


if __name__ == "__main__":
    main()