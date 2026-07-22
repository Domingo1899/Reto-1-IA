from pathlib import Path

import pandas as pd


BASE = Path("data/interim/entrega-02-corregida")

for anio in [2025, 2026]:
    ruta = BASE / f"docentes_{anio}.parquet"
    df = pd.read_parquet(ruta)

    print("\n" + "=" * 80)
    print(f"DOCENTES {anio}")
    print("=" * 80)

    print("\nCOLUMNAS:")
    print(df.columns.tolist())

    for columna in [
        "Rubro",
        "tipo_centro",
        "ciclo",
        "ID_CENTRO",
    ]:
        if columna in df.columns:
            print(f"\n--- {columna} ---")
            print(
                df[columna]
                .astype("string")
                .str.strip()
                .str.lower()
                .value_counts(dropna=False)
                .head(30)
                .to_string()
            )

    if (
        "ID_persona" in df.columns
        and "ID_CENTRO" in df.columns
    ):
        centros_por_docente = (
            df[["ID_persona", "ID_CENTRO"]]
            .dropna()
            .drop_duplicates()
            .groupby("ID_persona")["ID_CENTRO"]
            .nunique()
        )

        print(
            "\nDocentes asociados a más de un centro:",
            int((centros_por_docente > 1).sum()),
        )

        print(
            "\nCantidad de centros por docente:"
        )

        print(
            centros_por_docente
            .value_counts()
            .sort_index()
            .head(20)
            .to_string()
        )