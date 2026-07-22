from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path("data/interim/entrega-02-corregida")

CSV_PERCENTILES = Path("resultado_percentiles_panel.csv")
CSV_UMBRALES = Path("resultado_umbrales_distribucion.csv")
CSV_BANDAS = Path("resultado_cambio_por_banda_2025.csv")

PNG_HISTOGRAMA = Path("histograma_dias_panel.png")
PNG_COLA = Path("curva_usuarios_intensivos_panel.png")


def convertir_dias(serie: pd.Series) -> pd.Series:
    return pd.to_numeric(
        serie.astype("string")
        .str.strip()
        .str.replace(",", ".", regex=False),
        errors="coerce",
    ).fillna(0)


def cargar_total_por_persona(anio: int) -> pd.DataFrame:
    ruta = BASE / f"estudiantes_{anio}.parquet"

    df = pd.read_parquet(
        ruta,
        columns=["ID_persona", "Dias4", "Dias5", "Dias6"],
    )

    df = df.dropna(subset=["ID_persona"]).copy()
    df["ID_persona"] = df["ID_persona"].astype("string")

    for columna in ["Dias4", "Dias5", "Dias6"]:
        df[columna] = convertir_dias(df[columna])

    # Orden conceptual correcto:
    # 1. sumar los meses en cada fila;
    # 2. tomar el máximo por persona porque el dato se repite por materia.
    df["dias_total"] = df["Dias4"] + df["Dias5"] + df["Dias6"]

    return (
        df.groupby("ID_persona", as_index=False)["dias_total"]
        .max()
    )


def main() -> None:
    datos_2025 = cargar_total_por_persona(2025)
    datos_2026 = cargar_total_por_persona(2026)

    panel = datos_2025.merge(
        datos_2026,
        on="ID_persona",
        how="inner",
        suffixes=("_2025", "_2026"),
        validate="one_to_one",
    )

    x25 = panel["dias_total_2025"]
    x26 = panel["dias_total_2026"]

    print(f"\nPersonas presentes en ambos años: {len(panel):,}")
    print(f"Máximo 2025: {x25.max():.2f}")
    print(f"Máximo 2026: {x26.max():.2f}")

    # 1. Percentiles
    percentiles = [0, 10, 25, 50, 75, 90, 95, 99, 100]

    tabla_percentiles = pd.DataFrame(
        {
            "percentil": percentiles,
            "dias_2025": [
                x25.quantile(p / 100) for p in percentiles
            ],
            "dias_2026": [
                x26.quantile(p / 100) for p in percentiles
            ],
        }
    )

    tabla_percentiles["diferencia_dias"] = (
        tabla_percentiles["dias_2026"]
        - tabla_percentiles["dias_2025"]
    )

    tabla_percentiles["variacion_pct"] = np.where(
        tabla_percentiles["dias_2025"] != 0,
        (
            tabla_percentiles["dias_2026"]
            / tabla_percentiles["dias_2025"]
            - 1
        )
        * 100,
        np.nan,
    )

    # 2. Proporción por encima de distintos umbrales
    umbrales = [1, 5, 10, 20, 30, 45, 60]
    filas_umbrales = []

    for umbral in umbrales:
        tasa25 = (x25 >= umbral).mean() * 100
        tasa26 = (x26 >= umbral).mean() * 100

        filas_umbrales.append(
            {
                "umbral_dias": umbral,
                "tasa_2025_pct": tasa25,
                "tasa_2026_pct": tasa26,
                "diferencia_pp": tasa26 - tasa25,
            }
        )

    tabla_umbrales = pd.DataFrame(filas_umbrales)

    # 3. Cambio según nivel inicial de uso
    panel["banda_2025"] = pd.cut(
        x25,
        bins=[-np.inf, 0, 4, 9, 19, 39, np.inf],
        labels=["0", "1-4", "5-9", "10-19", "20-39", "40 o más"],
    )

    filas_bandas = []

    for banda, grupo in panel.groupby(
        "banda_2025",
        observed=True,
    ):
        media25 = grupo["dias_total_2025"].mean()
        media26 = grupo["dias_total_2026"].mean()
        cambio = (
            grupo["dias_total_2026"]
            - grupo["dias_total_2025"]
        )

        filas_bandas.append(
            {
                "banda_dias_2025": str(banda),
                "personas": len(grupo),
                "promedio_2025": media25,
                "promedio_2026": media26,
                "diferencia_promedio": media26 - media25,
                "variacion_promedio_pct": (
                    (media26 / media25 - 1) * 100
                    if media25 != 0
                    else np.nan
                ),
                "bajaron_pct": (cambio < 0).mean() * 100,
                "iguales_pct": (cambio == 0).mean() * 100,
                "subieron_pct": (cambio > 0).mean() * 100,
            }
        )

    tabla_bandas = pd.DataFrame(filas_bandas)

    # CSV
    tabla_percentiles.to_csv(
        CSV_PERCENTILES,
        sep=";",
        decimal=",",
        index=False,
    )

    tabla_umbrales.to_csv(
        CSV_UMBRALES,
        sep=";",
        decimal=",",
        index=False,
    )

    tabla_bandas.to_csv(
        CSV_BANDAS,
        sep=";",
        decimal=",",
        index=False,
    )

    # Histograma
    bins = np.arange(-0.5, 93, 3)

    plt.figure(figsize=(10, 6))
    plt.hist(
        x25,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2,
        label="2025",
    )
    plt.hist(
        x26,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2,
        label="2026",
    )
    plt.xlabel("Días de uso entre abril y junio")
    plt.ylabel("Densidad")
    plt.title("Distribución de días de uso de CREA")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PNG_HISTOGRAMA, dpi=180)
    plt.close()

    # Curva de supervivencia: porcentaje que alcanza cada cantidad de días.
    dias = np.arange(0, 92)

    supervivencia25 = [
        (x25 >= dia).mean() * 100 for dia in dias
    ]
    supervivencia26 = [
        (x26 >= dia).mean() * 100 for dia in dias
    ]

    plt.figure(figsize=(10, 6))
    plt.plot(dias, supervivencia25, label="2025")
    plt.plot(dias, supervivencia26, label="2026")
    plt.xlabel("Cantidad mínima de días")
    plt.ylabel("Personas que alcanzan el umbral (%)")
    plt.title("Cola de usuarios según intensidad")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PNG_COLA, dpi=180)
    plt.close()

    print("\nPERCENTILES")
    print(
        tabla_percentiles.to_string(
            index=False,
            float_format=lambda valor: f"{valor:.2f}",
        )
    )

    print("\nTASAS POR UMBRAL")
    print(
        tabla_umbrales.to_string(
            index=False,
            float_format=lambda valor: f"{valor:.2f}",
        )
    )

    print("\nCAMBIO SEGÚN INTENSIDAD EN 2025")
    print(
        tabla_bandas.to_string(
            index=False,
            float_format=lambda valor: f"{valor:.2f}",
        )
    )

    print("\nArchivos generados:")
    print(CSV_PERCENTILES)
    print(CSV_UMBRALES)
    print(CSV_BANDAS)
    print(PNG_HISTOGRAMA)
    print(PNG_COLA)


if __name__ == "__main__":
    main()