from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


BASE = Path("data/interim/entrega-02-corregida")

SALIDA_CENTROS = Path(
    "resultado_prediccion_docente_centros.csv"
)
SALIDA_REGRESIONES = Path(
    "resultado_prediccion_docente_regresiones.csv"
)
SALIDA_CUARTILES = Path(
    "resultado_prediccion_docente_cuartiles.csv"
)

MAPEO_RUBRO = {
    "dges": "Secundaria",
    "dgetp": "UTU",
}

MIN_ESTUDIANTES = 30
MIN_DOCENTES_TODOS = 5
MIN_DOCENTES_UN_CENTRO = 3


def convertir_dias(serie: pd.Series) -> pd.Series:
    return pd.to_numeric(
        serie.astype("string")
        .str.strip()
        .str.replace(",", ".", regex=False),
        errors="coerce",
    ).fillna(0)


def normalizar_texto(serie: pd.Series) -> pd.Series:
    return (
        serie.astype("string")
        .str.strip()
        .str.lower()
    )


def promedio_ponderado(
    valores: pd.Series,
    pesos: pd.Series,
) -> float:
    mascara = (
        valores.notna()
        & pesos.notna()
        & (pesos > 0)
    )

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
            "dept_nombre",
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
    df["Rubro"] = normalizar_texto(df["Rubro"])

    df["dept_nombre"] = (
        normalizar_texto(df["dept_nombre"])
        .fillna("sin dato")
    )

    df = df[
        df["Rubro"].isin(MAPEO_RUBRO)
    ].copy()

    for columna in ["Dias4", "Dias5", "Dias6"]:
        df[columna] = convertir_dias(df[columna])

    # Ya verificamos que cada estudiante tiene
    # un único centro y rubro dentro del año.
    personas = (
        df.groupby("ID_persona", as_index=False)
        .agg(
            ID_CENTRO=("ID_CENTRO", "first"),
            Rubro=("Rubro", "first"),
            dept_nombre=("dept_nombre", "first"),
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
    df["Rubro"] = normalizar_texto(df["Rubro"])

    df = df[
        df["Rubro"].isin(MAPEO_RUBRO)
    ].copy()

    for columna in ["Dias4", "Dias5", "Dias6"]:
        df[columna] = convertir_dias(df[columna])

    # Cantidad de centros en los que aparece cada docente.
    centros_por_docente = (
        df[["ID_persona", "ID_CENTRO"]]
        .drop_duplicates()
        .groupby("ID_persona")
        .size()
        .rename("cantidad_centros")
        .reset_index()
    )

    # Los días se repiten por materia.
    # El docente sí puede trabajar en varios centros.
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

    docentes = docentes.merge(
        centros_por_docente,
        on="ID_persona",
        how="left",
        validate="many_to_one",
    )

    docentes["subsistema"] = docentes["Rubro"].map(
        MAPEO_RUBRO
    )

    return docentes


def panel_estudiantes() -> pd.DataFrame:
    datos_2025 = cargar_estudiantes(2025)
    datos_2026 = cargar_estudiantes(2026)

    panel = datos_2025.merge(
        datos_2026,
        on="ID_persona",
        how="inner",
        suffixes=("_2025", "_2026"),
        validate="one_to_one",
    )

    # Misma persona, mismo centro y mismo subsistema.
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
    panel["departamento"] = panel["dept_nombre_2025"]

    panel["est_abril_2025"] = panel["Dias4_2025"]
    panel["est_abril_2026"] = panel["Dias4_2026"]

    panel["est_mj_2025"] = (
        panel["Dias5_2025"]
        + panel["Dias6_2025"]
    )

    panel["est_mj_2026"] = (
        panel["Dias5_2026"]
        + panel["Dias6_2026"]
    )

    return panel


def panel_docentes() -> pd.DataFrame:
    datos_2025 = cargar_docentes(2025)
    datos_2026 = cargar_docentes(2026)

    # Mismo docente, mismo centro y mismo subsistema.
    panel = datos_2025.merge(
        datos_2026,
        on=["ID_persona", "ID_CENTRO", "Rubro"],
        how="inner",
        suffixes=("_2025", "_2026"),
        validate="one_to_one",
    )

    panel["subsistema"] = panel["Rubro"].map(
        MAPEO_RUBRO
    )

    return panel


def resumir_estudiantes(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    return (
        panel.groupby(
            [
                "ID_CENTRO",
                "Rubro",
                "subsistema",
                "departamento",
            ],
            as_index=False,
        )
        .agg(
            estudiantes_panel=(
                "ID_persona",
                "nunique",
            ),
            est_abril_2025=(
                "est_abril_2025",
                "mean",
            ),
            est_abril_2026=(
                "est_abril_2026",
                "mean",
            ),
            est_mj_2025=(
                "est_mj_2025",
                "mean",
            ),
            est_mj_2026=(
                "est_mj_2026",
                "mean",
            ),
        )
    )


def resumir_docentes(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    return (
        panel.groupby(
            ["ID_CENTRO", "Rubro", "subsistema"],
            as_index=False,
        )
        .agg(
            docentes_panel=(
                "ID_persona",
                "nunique",
            ),
            doc_abril_2025=(
                "Dias4_2025",
                "mean",
            ),
            doc_abril_2026=(
                "Dias4_2026",
                "mean",
            ),
        )
    )


def construir_centros(
    estudiantes: pd.DataFrame,
    docentes: pd.DataFrame,
    universo: str,
    minimo_docentes: int,
) -> pd.DataFrame:
    centros_docentes = resumir_docentes(docentes)

    centros = estudiantes.merge(
        centros_docentes,
        on=["ID_CENTRO", "Rubro", "subsistema"],
        how="inner",
        validate="one_to_one",
    )

    centros = centros[
        (
            centros["estudiantes_panel"]
            >= MIN_ESTUDIANTES
        )
        & (
            centros["docentes_panel"]
            >= minimo_docentes
        )
    ].copy()

    centros["universo_docente"] = universo

    # Predictor: cambio docente ocurrido en abril.
    centros["delta_doc_abril"] = (
        centros["doc_abril_2026"]
        - centros["doc_abril_2025"]
    )

    # Control: cambio estudiantil que ya existía en abril.
    centros["delta_est_abril"] = (
        centros["est_abril_2026"]
        - centros["est_abril_2025"]
    )

    # Resultado posterior: cambio estudiantil de mayo-junio.
    centros["delta_est_mj"] = (
        centros["est_mj_2026"]
        - centros["est_mj_2025"]
    )

    return centros


def ejecutar_regresiones(
    centros: pd.DataFrame,
) -> pd.DataFrame:
    modelos = {
        "1_bruto": (
            "delta_est_mj ~ delta_doc_abril"
        ),
        "2_controla_caida_estudiantil_abril": (
            "delta_est_mj ~ "
            "delta_doc_abril + delta_est_abril"
        ),
        "3_ajustado_completo": (
            "delta_est_mj ~ "
            "delta_doc_abril + "
            "delta_est_abril + "
            "est_mj_2025 + "
            "doc_abril_2025 + "
            "C(departamento)"
        ),
    }

    resultados = []

    for (
        universo,
        subsistema,
    ), grupo in centros.groupby(
        ["universo_docente", "subsistema"]
    ):
        pearson = grupo["delta_doc_abril"].corr(
            grupo["delta_est_mj"],
            method="pearson",
        )

        spearman = grupo["delta_doc_abril"].corr(
            grupo["delta_est_mj"],
            method="spearman",
        )

        for nombre_modelo, formula in modelos.items():
            modelo = smf.wls(
                formula=formula,
                data=grupo,
                weights=grupo["estudiantes_panel"],
            ).fit(cov_type="HC1")

            intervalo = modelo.conf_int().loc[
                "delta_doc_abril"
            ]

            beta = modelo.params["delta_doc_abril"]
            desviacion_x = grupo[
                "delta_doc_abril"
            ].std()

            desviacion_y = grupo[
                "delta_est_mj"
            ].std()

            beta_estandarizado = (
                beta * desviacion_x / desviacion_y
                if desviacion_y != 0
                else np.nan
            )

            resultados.append(
                {
                    "universo_docente": universo,
                    "subsistema": subsistema,
                    "modelo": nombre_modelo,
                    "centros": len(grupo),
                    "estudiantes_representados": int(
                        grupo["estudiantes_panel"].sum()
                    ),
                    "pearson_bruto": pearson,
                    "spearman_bruto": spearman,
                    "beta_delta_doc_abril": beta,
                    "beta_estandarizado": (
                        beta_estandarizado
                    ),
                    "error_estandar_robusto": (
                        modelo.bse["delta_doc_abril"]
                    ),
                    "IC95_inferior": intervalo.iloc[0],
                    "IC95_superior": intervalo.iloc[1],
                    "p_valor": (
                        modelo.pvalues["delta_doc_abril"]
                    ),
                    "r_cuadrado": modelo.rsquared,
                }
            )

    return pd.DataFrame(resultados)


def crear_cuartiles(
    centros: pd.DataFrame,
) -> pd.DataFrame:
    resultados = []

    etiquetas = [
        "Q1 mayor caída docente",
        "Q2",
        "Q3",
        "Q4 mejor evolución docente",
    ]

    for (
        universo,
        subsistema,
    ), grupo in centros.groupby(
        ["universo_docente", "subsistema"]
    ):
        grupo = grupo.copy()

        grupo["cuartil_docente_abril"] = pd.qcut(
            grupo["delta_doc_abril"].rank(
                method="first"
            ),
            q=4,
            labels=etiquetas,
        )

        for cuartil, subgrupo in grupo.groupby(
            "cuartil_docente_abril",
            observed=True,
        ):
            resultados.append(
                {
                    "universo_docente": universo,
                    "subsistema": subsistema,
                    "cuartil_docente_abril": str(cuartil),
                    "centros": len(subgrupo),
                    "estudiantes_representados": int(
                        subgrupo[
                            "estudiantes_panel"
                        ].sum()
                    ),
                    "delta_doc_abril_promedio": (
                        promedio_ponderado(
                            subgrupo["delta_doc_abril"],
                            subgrupo["docentes_panel"],
                        )
                    ),
                    "delta_est_abril_promedio": (
                        promedio_ponderado(
                            subgrupo["delta_est_abril"],
                            subgrupo[
                                "estudiantes_panel"
                            ],
                        )
                    ),
                    "delta_est_mayo_junio_promedio": (
                        promedio_ponderado(
                            subgrupo["delta_est_mj"],
                            subgrupo[
                                "estudiantes_panel"
                            ],
                        )
                    ),
                }
            )

    return pd.DataFrame(resultados)


def main() -> None:
    estudiantes_persona = panel_estudiantes()
    docentes_persona = panel_docentes()

    estudiantes_centro = resumir_estudiantes(
        estudiantes_persona
    )

    # Universo 1: todos los pares docente-centro estables.
    centros_todos = construir_centros(
        estudiantes=estudiantes_centro,
        docentes=docentes_persona,
        universo="todos_docentes_estables",
        minimo_docentes=MIN_DOCENTES_TODOS,
    )

    # Universo 2: solamente docentes que tenían un centro
    # en ambos años. Es la prueba de robustez principal.
    docentes_un_centro = docentes_persona[
        (
            docentes_persona["cantidad_centros_2025"]
            == 1
        )
        & (
            docentes_persona["cantidad_centros_2026"]
            == 1
        )
    ].copy()

    centros_un_centro = construir_centros(
        estudiantes=estudiantes_centro,
        docentes=docentes_un_centro,
        universo="solo_docentes_un_centro",
        minimo_docentes=MIN_DOCENTES_UN_CENTRO,
    )

    centros = pd.concat(
        [centros_todos, centros_un_centro],
        ignore_index=True,
    )

    regresiones = ejecutar_regresiones(centros)
    cuartiles = crear_cuartiles(centros)

    centros.to_csv(
        SALIDA_CENTROS,
        sep=";",
        decimal=",",
        index=False,
    )

    regresiones.to_csv(
        SALIDA_REGRESIONES,
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

    print("\n" + "=" * 110)
    print("REGRESIONES: DOCENTE EN ABRIL → ESTUDIANTE EN MAYO-JUNIO")
    print("=" * 110)

    columnas_regresion = [
        "universo_docente",
        "subsistema",
        "modelo",
        "centros",
        "pearson_bruto",
        "spearman_bruto",
        "beta_delta_doc_abril",
        "IC95_inferior",
        "IC95_superior",
        "p_valor",
        "r_cuadrado",
    ]

    print(
        regresiones[columnas_regresion].to_string(
            index=False,
            float_format=lambda valor: f"{valor:.4f}",
        )
    )

    print("\n" + "=" * 110)
    print("CUARTILES SEGÚN CAMBIO DOCENTE EN ABRIL")
    print("=" * 110)

    print(
        cuartiles.to_string(
            index=False,
            float_format=lambda valor: f"{valor:.3f}",
        )
    )

    print("\nInterpretación del coeficiente:")
    print(
        "Un beta positivo indica que una caída de 1 día "
        "en la actividad docente de abril se asocia con "
        "una caída posterior de beta días en la actividad "
        "estudiantil acumulada de mayo-junio."
    )

    print("\nArchivos generados:")
    print(SALIDA_CENTROS)
    print(SALIDA_REGRESIONES)
    print(SALIDA_CUARTILES)


if __name__ == "__main__":
    main()