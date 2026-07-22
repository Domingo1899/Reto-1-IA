from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path("data/interim/entrega-02-corregida")

CSV_RESULTADOS = Path("resultado_subsistemas_panel.csv")
CSV_TRANSICIONES = Path("transiciones_subsistema_panel.csv")

MAPEO_RUBRO = {
    "dgeip": "Primaria",
    "dges": "Secundaria",
    "dgetp": "UTU",
}


def convertir_dias(serie: pd.Series) -> pd.Series:
    return pd.to_numeric(
        serie.astype("string")
        .str.strip()
        .str.replace(",", ".", regex=False),
        errors="coerce",
    ).fillna(0)


def cargar_por_persona(anio: int) -> pd.DataFrame:
    ruta = BASE / f"estudiantes_{anio}.parquet"

    df = pd.read_parquet(
        ruta,
        columns=[
            "ID_persona",
            "Rubro",
            "Dias4",
            "Dias5",
            "Dias6",
        ],
    )

    df = df.dropna(subset=["ID_persona", "Rubro"]).copy()
    df["ID_persona"] = df["ID_persona"].astype("string")
    df["Rubro"] = (
        df["Rubro"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    valores_desconocidos = (
        set(df["Rubro"].dropna().unique())
        - set(MAPEO_RUBRO)
    )

    if valores_desconocidos:
        raise ValueError(
            f"Valores de Rubro no contemplados: "
            f"{sorted(valores_desconocidos)}"
        )

    for columna in ["Dias4", "Dias5", "Dias6"]:
        df[columna] = convertir_dias(df[columna])

    df["dias_total"] = (
        df["Dias4"]
        + df["Dias5"]
        + df["Dias6"]
    )

    rubros_por_persona = (
        df[["ID_persona", "Rubro"]]
        .drop_duplicates()
        .groupby("ID_persona")["Rubro"]
        .nunique()
    )

    if (rubros_por_persona > 1).any():
        raise ValueError(
            f"Hay {(rubros_por_persona > 1).sum()} personas "
            "con más de un Rubro dentro del mismo año."
        )

    resultado = (
        df.groupby("ID_persona", as_index=False)
        .agg(
            dias_total=("dias_total", "max"),
            Rubro=("Rubro", "first"),
        )
    )

    resultado["subsistema"] = resultado["Rubro"].map(
        MAPEO_RUBRO
    )

    return resultado[
        ["ID_persona", "subsistema", "dias_total"]
    ]


def resumir(
    datos: pd.DataFrame,
    universo: str,
    columna_subsistema: str,
) -> list[dict]:
    filas = []

    for subsistema, grupo in datos.groupby(
        columna_subsistema
    ):
        media25 = grupo["dias_total_2025"].mean()
        media26 = grupo["dias_total_2026"].mean()

        for umbral in [1, 5, 10]:
            tasa25 = (
                grupo["dias_total_2025"] >= umbral
            ).mean() * 100

            tasa26 = (
                grupo["dias_total_2026"] >= umbral
            ).mean() * 100

            filas.append(
                {
                    "universo": universo,
                    "subsistema": subsistema,
                    "personas_panel": len(grupo),
                    "umbral_dias": umbral,
                    "tasa_2025_pct": tasa25,
                    "tasa_2026_pct": tasa26,
                    "diferencia_pp": tasa26 - tasa25,
                    "promedio_dias_2025": media25,
                    "promedio_dias_2026": media26,
                    "diferencia_promedio": media26 - media25,
                    "variacion_promedio_pct": (
                        (media26 / media25 - 1) * 100
                        if media25 != 0
                        else np.nan
                    ),
                }
            )

    return filas


def main() -> None:
    datos_2025 = cargar_por_persona(2025)
    datos_2026 = cargar_por_persona(2026)

    panel = datos_2025.merge(
        datos_2026,
        on="ID_persona",
        how="inner",
        suffixes=("_2025", "_2026"),
        validate="one_to_one",
    )

    # Transiciones entre subsistemas.
    transiciones = (
        panel.groupby(
            ["subsistema_2025", "subsistema_2026"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "personas"})
    )

    transiciones["porcentaje_panel"] = (
        transiciones["personas"] / len(panel) * 100
    )

    # Comparación limpia: permanecen en el mismo subsistema.
    panel_estable = panel[
        panel["subsistema_2025"]
        == panel["subsistema_2026"]
    ].copy()

    filas = []

    filas.extend(
        resumir(
            panel_estable,
            universo="panel_estable",
            columna_subsistema="subsistema_2025",
        )
    )

    # Análisis secundario: clasificación según subsistema de origen.
    filas.extend(
        resumir(
            panel,
            universo="origen_2025",
            columna_subsistema="subsistema_2025",
        )
    )

    resultado = pd.DataFrame(filas)

    resultado.to_csv(
        CSV_RESULTADOS,
        sep=";",
        decimal=",",
        index=False,
    )

    transiciones.to_csv(
        CSV_TRANSICIONES,
        sep=";",
        decimal=",",
        index=False,
    )

    print(f"\nPersonas presentes en ambos años: {len(panel):,}")
    print(
        "Personas que permanecen en el mismo subsistema:",
        f"{len(panel_estable):,}",
        f"({len(panel_estable) / len(panel) * 100:.2f}%)",
    )

    print("\nTRANSICIONES")
    print(
        transiciones.to_string(
            index=False,
            float_format=lambda valor: f"{valor:.2f}",
        )
    )

    print("\nRESULTADOS POR SUBSISTEMA")
    print(
        resultado.to_string(
            index=False,
            float_format=lambda valor: f"{valor:.2f}",
        )
    )

    print("\nArchivos generados:")
    print(CSV_RESULTADOS)
    print(CSV_TRANSICIONES)


if __name__ == "__main__":
    main()