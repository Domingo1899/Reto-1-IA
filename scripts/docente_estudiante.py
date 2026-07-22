from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path("data/interim/entrega-02-corregida")

SALIDA_CENTROS = Path(
    "resultado_docente_estudiante_centros.csv"
)
SALIDA_RESUMEN = Path(
    "resultado_docente_estudiante_resumen.csv"
)
SALIDA_CUARTILES = Path(
    "resultado_docente_estudiante_cuartiles.csv"
)

MAPEO_RUBRO = {
    "dges": "Secundaria",
    "dgetp": "UTU",
}

PERIODOS = {
    "Abril": ["Dias4"],
    "Mayo": ["Dias5"],
    "Junio": ["Dias6"],
    "Mayo-Junio": ["Dias5", "Dias6"],
}

MIN_ESTUDIANTES = 30
MIN_DOCENTES_PANEL = 5


def convertir_dias(serie: pd.Series) -> pd.Series:
    return pd.to_numeric(
        serie.astype("string")
        .str.strip()
        .str.replace(",", ".", regex=False),
        errors="coerce",
    ).fillna(0)


def normalizar(serie: pd.Series) -> pd.Series:
    return (
        serie.astype("string")
        .str.strip()
        .str.lower()
    )


def promedio_ponderado(
    valores: pd.Series,
    pesos: pd.Series,
) -> float:
    mascara = valores.notna() & pesos.notna() & (pesos > 0)

    if not mascara.any():
        return np.nan

    return float(
        np.average(
            valores[mascara],
            weights=pesos[mascara],
        )
    )


def cargar_estudiantes(anio: int) -> pd.DataFrame:
    ruta = BASE / f"estudiantes_{anio}.parquet"

    df = pd.read_parquet(
        ruta,
        columns=[
            "ID_persona",
            "ID_CENTRO",
            "Rubro",
            "Dias4",
            "Dias5",
            "Dias6",
        ],
    )

    df = df.dropna(
        subset=["ID_persona", "ID_CENTRO", "Rubro"]
    ).copy()

    df["ID_persona"] = df["ID_persona"].astype("string")
    df["ID_CENTRO"] = df["ID_CENTRO"].astype("string")
    df["Rubro"] = normalizar(df["Rubro"])

    df = df[df["Rubro"].isin(MAPEO_RUBRO)].copy()

    for columna in ["Dias4", "Dias5", "Dias6"]:
        df[columna] = convertir_dias(df[columna])

    # Verificar que una persona tenga un único centro y rubro.
    atributos = (
        df[["ID_persona", "ID_CENTRO", "Rubro"]]
        .drop_duplicates()
    )

    cantidad_atributos = (
        atributos.groupby("ID_persona")
        .size()
    )

    ambiguos = int((cantidad_atributos > 1).sum())

    if ambiguos > 0:
        raise ValueError(
            f"Hay {ambiguos} estudiantes asociados a más "
            "de una combinación centro-rubro."
        )

    # Los días están repetidos por materia.
    personas = (
        df.groupby("ID_persona", as_index=False)
        .agg(
            ID_CENTRO=("ID_CENTRO", "first"),
            Rubro=("Rubro", "first"),
            Dias4=("Dias4", "max"),
            Dias5=("Dias5", "max"),
            Dias6=("Dias6", "max"),
        )
    )

    personas["subsistema"] = personas["Rubro"].map(
        MAPEO_RUBRO
    )

    return personas


def cargar_docentes(anio: int) -> pd.DataFrame:
    ruta = BASE / f"docentes_{anio}.parquet"

    df = pd.read_parquet(
        ruta,
        columns=[
            "ID_persona",
            "ID_CENTRO",
            "Rubro",
            "Dias4",
            "Dias5",
            "Dias6",
        ],
    )

    df = df.dropna(
        subset=["ID_persona", "ID_CENTRO", "Rubro"]
    ).copy()

    df["ID_persona"] = df["ID_persona"].astype("string")
    df["ID_CENTRO"] = df["ID_CENTRO"].astype("string")
    df["Rubro"] = normalizar(df["Rubro"])

    df = df[df["Rubro"].isin(MAPEO_RUBRO)].copy()

    for columna in ["Dias4", "Dias5", "Dias6"]:
        df[columna] = convertir_dias(df[columna])

    # Un docente puede trabajar en varios centros.
    # Se deja una fila por docente-centro.
    docentes = (
        df.groupby(
            ["ID_persona", "ID_CENTRO", "Rubro"],
            as_index=False,
        )
        .agg(
            Dias4=("Dias4", "max"),
            Dias5=("Dias5", "max"),
            Dias6=("Dias6", "max"),
        )
    )

    docentes["subsistema"] = docentes["Rubro"].map(
        MAPEO_RUBRO
    )

    return docentes


def construir_panel_estudiantes(
    estudiantes_2025: pd.DataFrame,
    estudiantes_2026: pd.DataFrame,
) -> pd.DataFrame:
    panel = estudiantes_2025.merge(
        estudiantes_2026,
        on="ID_persona",
        how="inner",
        suffixes=("_2025", "_2026"),
        validate="one_to_one",
    )

    # Mismos estudiantes, mismo centro y mismo subsistema.
    panel = panel[
        (
            panel["ID_CENTRO_2025"]
            == panel["ID_CENTRO_2026"]
        )
        & (
            panel["Rubro_2025"]
            == panel["Rubro_2026"]
        )
    ].copy()

    panel["ID_CENTRO"] = panel["ID_CENTRO_2025"]
    panel["Rubro"] = panel["Rubro_2025"]
    panel["subsistema"] = panel["subsistema_2025"]

    return panel


def construir_panel_docentes(
    docentes_2025: pd.DataFrame,
    docentes_2026: pd.DataFrame,
) -> pd.DataFrame:
    # Mismo docente, mismo centro y mismo rubro.
    panel = docentes_2025.merge(
        docentes_2026,
        on=["ID_persona", "ID_CENTRO", "Rubro"],
        how="inner",
        suffixes=("_2025", "_2026"),
        validate="one_to_one",
    )

    panel["subsistema"] = panel["Rubro"].map(
        MAPEO_RUBRO
    )

    return panel


def calcular_retencion_docente(
    docentes_2025: pd.DataFrame,
    docentes_2026: pd.DataFrame,
) -> pd.DataFrame:
    claves = ["ID_persona", "ID_CENTRO", "Rubro"]

    pares_2025 = docentes_2025[claves].drop_duplicates()
    pares_2026 = docentes_2026[claves].drop_duplicates()

    retenidos = pares_2025.merge(
        pares_2026,
        on=claves,
        how="inner",
        validate="one_to_one",
    )

    cantidad_2025 = (
        pares_2025.groupby(
            ["ID_CENTRO", "Rubro"],
            as_index=False,
        )
        .agg(docentes_2025=("ID_persona", "nunique"))
    )

    cantidad_2026 = (
        pares_2026.groupby(
            ["ID_CENTRO", "Rubro"],
            as_index=False,
        )
        .agg(docentes_2026=("ID_persona", "nunique"))
    )

    cantidad_retenidos = (
        retenidos.groupby(
            ["ID_CENTRO", "Rubro"],
            as_index=False,
        )
        .agg(
            docentes_retenidos=(
                "ID_persona",
                "nunique",
            )
        )
    )

    resultado = (
        cantidad_2025
        .merge(
            cantidad_2026,
            on=["ID_CENTRO", "Rubro"],
            how="outer",
        )
        .merge(
            cantidad_retenidos,
            on=["ID_CENTRO", "Rubro"],
            how="left",
        )
        .fillna(0)
    )

    resultado["retencion_docente_pct"] = np.where(
        resultado["docentes_2025"] > 0,
        (
            resultado["docentes_retenidos"]
            / resultado["docentes_2025"]
            * 100
        ),
        np.nan,
    )

    resultado["variacion_cantidad_docentes_pct"] = np.where(
        resultado["docentes_2025"] > 0,
        (
            resultado["docentes_2026"]
            / resultado["docentes_2025"]
            - 1
        )
        * 100,
        np.nan,
    )

    return resultado


def agregar_periodo(
    df: pd.DataFrame,
    columnas: list[str],
    sufijo: str,
) -> pd.Series:
    return sum(
        df[f"{columna}_{sufijo}"]
        for columna in columnas
    )


def resumir_estudiantes_por_centro(
    panel: pd.DataFrame,
    periodo: str,
    columnas: list[str],
) -> pd.DataFrame:
    temporal = panel.copy()

    temporal["valor_2025"] = agregar_periodo(
        temporal,
        columnas,
        "2025",
    )

    temporal["valor_2026"] = agregar_periodo(
        temporal,
        columnas,
        "2026",
    )

    resultado = (
        temporal.groupby(
            ["ID_CENTRO", "Rubro", "subsistema"],
            as_index=False,
        )
        .agg(
            estudiantes_panel=("ID_persona", "nunique"),
            promedio_estudiantes_2025=(
                "valor_2025",
                "mean",
            ),
            promedio_estudiantes_2026=(
                "valor_2026",
                "mean",
            ),
            tasa5_estudiantes_2025=(
                "valor_2025",
                lambda x: (x >= 5).mean() * 100,
            ),
            tasa5_estudiantes_2026=(
                "valor_2026",
                lambda x: (x >= 5).mean() * 100,
            ),
        )
    )

    resultado["periodo"] = periodo

    resultado["delta_estudiantes"] = (
        resultado["promedio_estudiantes_2026"]
        - resultado["promedio_estudiantes_2025"]
    )

    resultado["delta_tasa5_estudiantes_pp"] = (
        resultado["tasa5_estudiantes_2026"]
        - resultado["tasa5_estudiantes_2025"]
    )

    return resultado


def resumir_docentes_por_centro(
    panel: pd.DataFrame,
    periodo: str,
    columnas: list[str],
) -> pd.DataFrame:
    temporal = panel.copy()

    temporal["valor_2025"] = agregar_periodo(
        temporal,
        columnas,
        "2025",
    )

    temporal["valor_2026"] = agregar_periodo(
        temporal,
        columnas,
        "2026",
    )

    resultado = (
        temporal.groupby(
            ["ID_CENTRO", "Rubro", "subsistema"],
            as_index=False,
        )
        .agg(
            docentes_panel=("ID_persona", "nunique"),
            promedio_docentes_2025=(
                "valor_2025",
                "mean",
            ),
            promedio_docentes_2026=(
                "valor_2026",
                "mean",
            ),
            tasa5_docentes_2025=(
                "valor_2025",
                lambda x: (x >= 5).mean() * 100,
            ),
            tasa5_docentes_2026=(
                "valor_2026",
                lambda x: (x >= 5).mean() * 100,
            ),
        )
    )

    resultado["periodo"] = periodo

    resultado["delta_docentes"] = (
        resultado["promedio_docentes_2026"]
        - resultado["promedio_docentes_2025"]
    )

    resultado["delta_tasa5_docentes_pp"] = (
        resultado["tasa5_docentes_2026"]
        - resultado["tasa5_docentes_2025"]
    )

    return resultado


def crear_resumen(
    centros: pd.DataFrame,
) -> pd.DataFrame:
    filas = []

    for (subsistema, periodo), grupo in centros.groupby(
        ["subsistema", "periodo"]
    ):
        docentes_bajan = grupo["delta_docentes"] < 0
        docentes_no_bajan = ~docentes_bajan

        promedio_est_si_bajan = promedio_ponderado(
            grupo.loc[
                docentes_bajan,
                "delta_estudiantes",
            ],
            grupo.loc[
                docentes_bajan,
                "estudiantes_panel",
            ],
        )

        promedio_est_si_no_bajan = promedio_ponderado(
            grupo.loc[
                docentes_no_bajan,
                "delta_estudiantes",
            ],
            grupo.loc[
                docentes_no_bajan,
                "estudiantes_panel",
            ],
        )

        filas.append(
            {
                "subsistema": subsistema,
                "periodo": periodo,
                "centros_analizados": len(grupo),

                "pearson_delta_promedio": (
                    grupo["delta_docentes"].corr(
                        grupo["delta_estudiantes"],
                        method="pearson",
                    )
                ),

                "spearman_delta_promedio": (
                    grupo["delta_docentes"].corr(
                        grupo["delta_estudiantes"],
                        method="spearman",
                    )
                ),

                "pearson_delta_tasa5": (
                    grupo[
                        "delta_tasa5_docentes_pp"
                    ].corr(
                        grupo[
                            "delta_tasa5_estudiantes_pp"
                        ],
                        method="pearson",
                    )
                ),

                "spearman_delta_tasa5": (
                    grupo[
                        "delta_tasa5_docentes_pp"
                    ].corr(
                        grupo[
                            "delta_tasa5_estudiantes_pp"
                        ],
                        method="spearman",
                    )
                ),

                "spearman_retencion_docente": (
                    grupo[
                        "retencion_docente_pct"
                    ].corr(
                        grupo["delta_estudiantes"],
                        method="spearman",
                    )
                ),

                "centros_docente_y_estudiante_bajan_pct": (
                    (
                        (grupo["delta_docentes"] < 0)
                        & (grupo["delta_estudiantes"] < 0)
                    ).mean()
                    * 100
                ),

                "delta_estudiantes_si_docentes_bajan": (
                    promedio_est_si_bajan
                ),

                "delta_estudiantes_si_docentes_no_bajan": (
                    promedio_est_si_no_bajan
                ),

                "diferencia_entre_grupos": (
                    promedio_est_si_bajan
                    - promedio_est_si_no_bajan
                ),

                "retencion_docente_promedio_pct": (
                    promedio_ponderado(
                        grupo["retencion_docente_pct"],
                        grupo["estudiantes_panel"],
                    )
                ),
            }
        )

    return pd.DataFrame(filas)


def crear_cuartiles(
    centros: pd.DataFrame,
) -> pd.DataFrame:
    filas = []

    for (subsistema, periodo), grupo in centros.groupby(
        ["subsistema", "periodo"]
    ):
        grupo = grupo.copy()

        if len(grupo) < 8:
            continue

        # Q1 = centros con mayor caída de actividad docente.
        grupo["cuartil_docente"] = pd.qcut(
            grupo["delta_docentes"].rank(
                method="first"
            ),
            q=4,
            labels=[
                "Q1 mayor caída docente",
                "Q2",
                "Q3",
                "Q4 mejor evolución docente",
            ],
        )

        for cuartil, subgrupo in grupo.groupby(
            "cuartil_docente",
            observed=True,
        ):
            filas.append(
                {
                    "subsistema": subsistema,
                    "periodo": periodo,
                    "cuartil_docente": str(cuartil),
                    "centros": len(subgrupo),

                    "delta_docente_promedio": (
                        promedio_ponderado(
                            subgrupo["delta_docentes"],
                            subgrupo["docentes_panel"],
                        )
                    ),

                    "delta_estudiante_promedio": (
                        promedio_ponderado(
                            subgrupo["delta_estudiantes"],
                            subgrupo["estudiantes_panel"],
                        )
                    ),

                    "delta_tasa5_estudiantes_pp": (
                        promedio_ponderado(
                            subgrupo[
                                "delta_tasa5_estudiantes_pp"
                            ],
                            subgrupo["estudiantes_panel"],
                        )
                    ),

                    "retencion_docente_promedio_pct": (
                        promedio_ponderado(
                            subgrupo[
                                "retencion_docente_pct"
                            ],
                            subgrupo["estudiantes_panel"],
                        )
                    ),
                }
            )

    return pd.DataFrame(filas)


def main() -> None:
    estudiantes_2025 = cargar_estudiantes(2025)
    estudiantes_2026 = cargar_estudiantes(2026)

    docentes_2025 = cargar_docentes(2025)
    docentes_2026 = cargar_docentes(2026)

    panel_estudiantes = construir_panel_estudiantes(
        estudiantes_2025,
        estudiantes_2026,
    )

    panel_docentes = construir_panel_docentes(
        docentes_2025,
        docentes_2026,
    )

    retencion_docente = calcular_retencion_docente(
        docentes_2025,
        docentes_2026,
    )

    resultados_centros = []

    for periodo, columnas in PERIODOS.items():
        estudiantes_centro = resumir_estudiantes_por_centro(
            panel_estudiantes,
            periodo,
            columnas,
        )

        docentes_centro = resumir_docentes_por_centro(
            panel_docentes,
            periodo,
            columnas,
        )

        combinado = estudiantes_centro.merge(
            docentes_centro,
            on=[
                "ID_CENTRO",
                "Rubro",
                "subsistema",
                "periodo",
            ],
            how="inner",
            validate="one_to_one",
        )

        combinado = combinado.merge(
            retencion_docente,
            on=["ID_CENTRO", "Rubro"],
            how="left",
            validate="many_to_one",
        )

        resultados_centros.append(combinado)

    centros = pd.concat(
        resultados_centros,
        ignore_index=True,
    )

    centros = centros[
        (centros["estudiantes_panel"] >= MIN_ESTUDIANTES)
        & (
            centros["docentes_panel"]
            >= MIN_DOCENTES_PANEL
        )
    ].copy()

    resumen = crear_resumen(centros)
    cuartiles = crear_cuartiles(centros)

    centros.to_csv(
        SALIDA_CENTROS,
        sep=";",
        decimal=",",
        index=False,
    )

    resumen.to_csv(
        SALIDA_RESUMEN,
        sep=";",
        decimal=",",
        index=False,
    )

    cuartiles.to_csv(
        SALIDA_CUARTILES,
        sep=";",
        decimal=",",
        index=False,
    )

    print("\nPANELES")
    print(
        "Estudiantes, mismo centro y subsistema:",
        f"{len(panel_estudiantes):,}",
    )
    print(
        "Docentes, mismo docente-centro:",
        f"{len(panel_docentes):,}",
    )

    print("\n" + "=" * 100)
    print("RELACIÓN DOCENTE-ESTUDIANTE")
    print("=" * 100)

    print(
        resumen.to_string(
            index=False,
            float_format=lambda valor: f"{valor:.3f}",
        )
    )

    print("\n" + "=" * 100)
    print("CUARTILES SEGÚN CAMBIO DOCENTE")
    print("=" * 100)

    print(
        cuartiles.to_string(
            index=False,
            float_format=lambda valor: f"{valor:.3f}",
        )
    )

    print("\nArchivos generados:")
    print(SALIDA_CENTROS)
    print(SALIDA_RESUMEN)
    print(SALIDA_CUARTILES)


if __name__ == "__main__":
    main()
    