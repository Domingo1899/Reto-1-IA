from pathlib import Path

import pandas as pd


BASE = Path("data/interim/entrega-02-corregida")
SALIDA = Path("resultado_mes_por_subsistema.csv")

MESES = {
    "Dias4": "Abril",
    "Dias5": "Mayo",
    "Dias6": "Junio",
}

SUBSISTEMAS = {
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


def cargar(anio: int) -> pd.DataFrame:
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

    for columna in MESES:
        df[columna] = convertir_dias(df[columna])

    rubros_por_persona = (
        df[["ID_persona", "Rubro"]]
        .drop_duplicates()
        .groupby("ID_persona")["Rubro"]
        .nunique()
    )

    if (rubros_por_persona > 1).any():
        raise ValueError(
            "Hay personas asociadas a más de un Rubro "
            "dentro del mismo año."
        )

    # Los días mensuales están repetidos por materia.
    personas = (
        df.groupby("ID_persona", as_index=False)
        .agg(
            Rubro=("Rubro", "first"),
            Dias4=("Dias4", "max"),
            Dias5=("Dias5", "max"),
            Dias6=("Dias6", "max"),
        )
    )

    personas["subsistema"] = personas["Rubro"].map(SUBSISTEMAS)

    if personas["subsistema"].isna().any():
        desconocidos = personas.loc[
            personas["subsistema"].isna(),
            "Rubro",
        ].unique()

        raise ValueError(
            f"Rubros desconocidos: {desconocidos}"
        )

    return personas


def main() -> None:
    datos_2025 = cargar(2025)
    datos_2026 = cargar(2026)

    panel = datos_2025.merge(
        datos_2026,
        on="ID_persona",
        how="inner",
        suffixes=("_2025", "_2026"),
        validate="one_to_one",
    )

    # Análisis principal: permanece en el mismo subsistema.
    panel = panel[
        panel["subsistema_2025"]
        == panel["subsistema_2026"]
    ].copy()

    resultados = []

    for subsistema, grupo in panel.groupby("subsistema_2025"):
        for columna, mes in MESES.items():
            x25 = grupo[f"{columna}_2025"]
            x26 = grupo[f"{columna}_2026"]

            activos_ambos = (x25 >= 1) & (x26 >= 1)

            resultados.append(
                {
                    "subsistema": subsistema,
                    "mes": mes,
                    "personas_panel": len(grupo),

                    "promedio_2025": x25.mean(),
                    "promedio_2026": x26.mean(),
                    "diferencia_promedio": (
                        x26.mean() - x25.mean()
                    ),
                    "variacion_promedio_pct": (
                        (x26.mean() / x25.mean() - 1) * 100
                        if x25.mean() != 0
                        else pd.NA
                    ),

                    "tasa_entro_2025_pct": (
                        (x25 >= 1).mean() * 100
                    ),
                    "tasa_entro_2026_pct": (
                        (x26 >= 1).mean() * 100
                    ),
                    "diferencia_acceso_pp": (
                        ((x26 >= 1).mean() - (x25 >= 1).mean())
                        * 100
                    ),

                    "tasa_5_dias_2025_pct": (
                        (x25 >= 5).mean() * 100
                    ),
                    "tasa_5_dias_2026_pct": (
                        (x26 >= 5).mean() * 100
                    ),
                    "diferencia_5_dias_pp": (
                        ((x26 >= 5).mean() - (x25 >= 5).mean())
                        * 100
                    ),

                    # Aísla intensidad entre quienes estuvieron
                    # activos en ese mes en ambos años.
                    "activos_ambos_anios": int(activos_ambos.sum()),
                    "promedio_activos_2025": (
                        x25[activos_ambos].mean()
                    ),
                    "promedio_activos_2026": (
                        x26[activos_ambos].mean()
                    ),
                    "diferencia_activos": (
                        x26[activos_ambos].mean()
                        - x25[activos_ambos].mean()
                    ),
                }
            )

    resultado = pd.DataFrame(resultados)

    orden_subsistema = {
        "Primaria": 1,
        "Secundaria": 2,
        "UTU": 3,
    }

    orden_mes = {
        "Abril": 1,
        "Mayo": 2,
        "Junio": 3,
    }

    resultado["_orden_subsistema"] = (
        resultado["subsistema"].map(orden_subsistema)
    )

    resultado["_orden_mes"] = resultado["mes"].map(orden_mes)

    resultado = (
        resultado.sort_values(
            ["_orden_subsistema", "_orden_mes"]
        )
        .drop(columns=["_orden_subsistema", "_orden_mes"])
    )

    resultado.to_csv(
        SALIDA,
        sep=";",
        decimal=",",
        index=False,
    )

    print(
        "\nPersonas en panel estable:",
        f"{len(panel):,}",
    )

    print("\nMES POR SUBSISTEMA")

    print(
        resultado.to_string(
            index=False,
            float_format=lambda valor: f"{valor:.2f}",
        )
    )

    print(f"\nCSV generado: {SALIDA}")


if __name__ == "__main__":
    main()