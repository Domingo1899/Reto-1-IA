from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path("data/interim/entrega-02-corregida")
ARCHIVO_SALIDA = Path("resultado_estacionalidad_panel.csv")

MESES = {
    "Dias4": "Abril",
    "Dias5": "Mayo",
    "Dias6": "Junio",
}


def convertir_dias(serie: pd.Series) -> pd.Series:
    return pd.to_numeric(
        serie.astype("string")
        .str.strip()
        .str.replace(",", ".", regex=False),
        errors="coerce",
    ).fillna(0)


def cargar_mes_por_persona(anio: int) -> pd.DataFrame:
    ruta = BASE / f"estudiantes_{anio}.parquet"

    df = pd.read_parquet(
        ruta,
        columns=["ID_persona", "Dias4", "Dias5", "Dias6"],
    )

    df = df.dropna(subset=["ID_persona"]).copy()
    df["ID_persona"] = df["ID_persona"].astype("string")

    for columna in MESES:
        df[columna] = convertir_dias(df[columna])

    # Cada mes está repetido por materia: se toma el máximo por persona.
    return (
        df.groupby("ID_persona", as_index=False)[list(MESES)]
        .max()
    )


def main() -> None:
    datos_2025 = cargar_mes_por_persona(2025)
    datos_2026 = cargar_mes_por_persona(2026)

    panel = datos_2025.merge(
        datos_2026,
        on="ID_persona",
        how="inner",
        suffixes=("_2025", "_2026"),
        validate="one_to_one",
    )

    caidas_promedio = {}

    for columna in MESES:
        media_2025 = panel[f"{columna}_2025"].mean()
        media_2026 = panel[f"{columna}_2026"].mean()
        caidas_promedio[columna] = media_2025 - media_2026

    caida_total = sum(caidas_promedio.values())
    resultados = []

    for columna, mes in MESES.items():
        x25 = panel[f"{columna}_2025"]
        x26 = panel[f"{columna}_2026"]
        cambio = x26 - x25

        aporte = (
            caidas_promedio[columna] / caida_total * 100
            if caida_total != 0
            else np.nan
        )

        resultados.append(
            {
                "mes": mes,
                "personas_panel": len(panel),
                "promedio_2025": x25.mean(),
                "promedio_2026": x26.mean(),
                "diferencia_promedio": x26.mean() - x25.mean(),
                "variacion_promedio_pct": (
                    (x26.mean() / x25.mean() - 1) * 100
                    if x25.mean() != 0
                    else np.nan
                ),
                "mediana_2025": x25.median(),
                "mediana_2026": x26.median(),
                "tasa_entro_2025_pct": (x25 >= 1).mean() * 100,
                "tasa_entro_2026_pct": (x26 >= 1).mean() * 100,
                "diferencia_acceso_pp": (
                    ((x26 >= 1).mean() - (x25 >= 1).mean()) * 100
                ),
                "personas_bajaron_pct": (cambio < 0).mean() * 100,
                "personas_iguales_pct": (cambio == 0).mean() * 100,
                "personas_subieron_pct": (cambio > 0).mean() * 100,
                "aporte_a_caida_total_pct": aporte,
            }
        )

    resultado = pd.DataFrame(resultados)

    resultado.to_csv(
        ARCHIVO_SALIDA,
        sep=";",
        decimal=",",
        index=False,
    )

    print(f"\nPersonas presentes en ambos años: {len(panel):,}")
    print("\nRESULTADO MENSUAL")
    print(
        resultado.to_string(
            index=False,
            float_format=lambda valor: f"{valor:.2f}",
        )
    )
    print(f"\nCSV generado: {ARCHIVO_SALIDA}")


if __name__ == "__main__":
    main()