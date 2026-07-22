from pathlib import Path

import pandas as pd


BASE = Path("data/interim/entrega-02-corregida")
COLUMNAS = ["ID_persona", "Rubro", "tipo_centro", "ciclo"]


def normalizar_texto(serie: pd.Series) -> pd.Series:
    return (
        serie.astype("string")
        .str.strip()
        .str.lower()
    )


for anio in (2025, 2026):
    ruta = BASE / f"estudiantes_{anio}.parquet"
    df = pd.read_parquet(ruta, columns=COLUMNAS)

    print("\n" + "=" * 80)
    print(f"ESTUDIANTES {anio}")
    print("=" * 80)

    # Versiones normalizadas solo para inspección.
    for columna in ["Rubro", "tipo_centro", "ciclo"]:
        df[f"{columna}_norm"] = normalizar_texto(df[columna])

        conteo_personas = (
            df[["ID_persona", f"{columna}_norm"]]
            .drop_duplicates()
            [f"{columna}_norm"]
            .value_counts(dropna=False)
        )

        print(f"\n--- {columna}: personas únicas ---")
        print(conteo_personas.to_string())

    # Comprobar si una persona aparece asociada a más de un Rubro.
    rubros_por_persona = (
        df[["ID_persona", "Rubro_norm"]]
        .drop_duplicates()
        .groupby("ID_persona")["Rubro_norm"]
        .nunique(dropna=False)
    )

    print("\n--- Cantidad de rubros diferentes por persona ---")
    print(rubros_por_persona.value_counts().sort_index().to_string())

    print(
        "\nPersonas asociadas a más de un Rubro:",
        int((rubros_por_persona > 1).sum()),
    )

    # Relación entre Rubro y tipo de centro.
    cruce = pd.crosstab(
        df.drop_duplicates(
            ["ID_persona", "Rubro_norm", "tipo_centro_norm"]
        )["Rubro_norm"],
        df.drop_duplicates(
            ["ID_persona", "Rubro_norm", "tipo_centro_norm"]
        )["tipo_centro_norm"],
        margins=True,
    )

    print("\n--- Cruce Rubro × tipo_centro, personas únicas ---")
    print(cruce.to_string())