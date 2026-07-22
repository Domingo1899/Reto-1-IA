from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path("data/interim/entrega-02-corregida")

SALIDA_RETENCION = Path("resultado_retencion_mensual.csv")
SALIDA_FOCOS = Path("resultado_focos_secundaria_utu.csv")
SALIDA_CENTROS = Path("resultado_centros_secundaria_utu.csv")
SALIDA_CONCENTRACION = Path("resultado_concentracion_centros.csv")

COLUMNAS = [
    "ID_persona",
    "ID_CENTRO",
    "Dias4",
    "Dias5",
    "Dias6",
    "Sexo",
    "ZONA",
    "tipo_centro",
    "Rubro",
    "dept_nombre",
    "ciclo",
    "grado",
    "CONTEXTO",
]

MAPEO_RUBRO = {
    "dges": "Secundaria",
    "dgetp": "UTU",
    "dgeip": "Primaria",
}


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


def mapa_valor_unico(
    df: pd.DataFrame,
    columna: str,
) -> tuple[pd.DataFrame, int]:
    """
    Devuelve una fila por persona solamente cuando la persona
    tiene un único valor no nulo para la columna.
    """

    temporal = df[["ID_persona", columna]].dropna().copy()

    temporal["ID_persona"] = temporal["ID_persona"].astype(
        "string"
    )

    temporal[columna] = normalizar_texto(
        temporal[columna]
    )

    temporal = temporal[
        temporal[columna].notna()
        & (temporal[columna] != "")
    ].drop_duplicates()

    cantidad_valores = (
        temporal.groupby("ID_persona")[columna]
        .nunique()
    )

    ids_validos = cantidad_valores[
        cantidad_valores == 1
    ].index

    ambiguos = int(
        (cantidad_valores > 1).sum()
    )

    resultado = (
        temporal[
            temporal["ID_persona"].isin(ids_validos)
        ]
        .drop_duplicates("ID_persona")
        [["ID_persona", columna]]
    )

    return resultado, ambiguos


def cargar_anio(
    anio: int,
) -> tuple[
    pd.DataFrame,
    dict[str, pd.DataFrame],
    dict[str, int],
]:
    ruta = BASE / f"estudiantes_{anio}.parquet"

    df = pd.read_parquet(
        ruta,
        columns=COLUMNAS,
    )

    df = df.dropna(
        subset=["ID_persona"]
    ).copy()

    df["ID_persona"] = df[
        "ID_persona"
    ].astype("string")

    for columna in ["Dias4", "Dias5", "Dias6"]:
        df[columna] = convertir_dias(
            df[columna]
        )

    # Los días están repetidos por materia.
    dias_persona = (
        df.groupby(
            "ID_persona",
            as_index=False,
        )[["Dias4", "Dias5", "Dias6"]]
        .max()
    )

    mapas = {}
    ambiguos = {}

    for columna in [
        "Rubro",
        "ID_CENTRO",
        "Sexo",
        "ZONA",
        "tipo_centro",
        "dept_nombre",
        "ciclo",
        "grado",
        "CONTEXTO",
    ]:
        mapa, cantidad_ambiguos = mapa_valor_unico(
            df,
            columna,
        )

        mapas[columna] = mapa
        ambiguos[columna] = cantidad_ambiguos

    personas = dias_persona.merge(
        mapas["Rubro"],
        on="ID_persona",
        how="inner",
        validate="one_to_one",
    )

    personas["subsistema"] = (
        personas["Rubro"]
        .map(MAPEO_RUBRO)
    )

    desconocidos = personas.loc[
        personas["subsistema"].isna(),
        "Rubro",
    ].unique()

    if len(desconocidos) > 0:
        raise ValueError(
            f"Rubros no contemplados: {desconocidos}"
        )

    return personas, mapas, ambiguos


def analizar_retencion(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compara la continuidad entre personas que alcanzaron
    el umbral de abril en ambos años.
    """

    resultados = []

    for subsistema, grupo in panel.groupby(
        "subsistema_2025"
    ):
        for umbral in [1, 5]:
            cohorte = grupo[
                (grupo["Dias4_2025"] >= umbral)
                & (grupo["Dias4_2026"] >= umbral)
            ].copy()

            for destino, columna in [
                ("Mayo", "Dias5"),
                ("Junio", "Dias6"),
            ]:
                tasa_2025 = (
                    cohorte[f"{columna}_2025"]
                    >= umbral
                ).mean() * 100

                tasa_2026 = (
                    cohorte[f"{columna}_2026"]
                    >= umbral
                ).mean() * 100

                promedio_2025 = cohorte[
                    f"{columna}_2025"
                ].mean()

                promedio_2026 = cohorte[
                    f"{columna}_2026"
                ].mean()

                resultados.append(
                    {
                        "subsistema": subsistema,
                        "umbral_abril": umbral,
                        "destino": destino,
                        "personas_activas_abril_ambos": len(
                            cohorte
                        ),
                        "retencion_2025_pct": tasa_2025,
                        "retencion_2026_pct": tasa_2026,
                        "diferencia_retencion_pp": (
                            tasa_2026 - tasa_2025
                        ),
                        "promedio_dias_2025": promedio_2025,
                        "promedio_dias_2026": promedio_2026,
                        "diferencia_promedio": (
                            promedio_2026
                            - promedio_2025
                        ),
                    }
                )

            # Continuidad completa: sigue activo en mayo y junio.
            continuo_2025 = (
                (cohorte["Dias5_2025"] >= umbral)
                & (cohorte["Dias6_2025"] >= umbral)
            ).mean() * 100

            continuo_2026 = (
                (cohorte["Dias5_2026"] >= umbral)
                & (cohorte["Dias6_2026"] >= umbral)
            ).mean() * 100

            resultados.append(
                {
                    "subsistema": subsistema,
                    "umbral_abril": umbral,
                    "destino": "Mayo y junio",
                    "personas_activas_abril_ambos": len(
                        cohorte
                    ),
                    "retencion_2025_pct": continuo_2025,
                    "retencion_2026_pct": continuo_2026,
                    "diferencia_retencion_pp": (
                        continuo_2026
                        - continuo_2025
                    ),
                    "promedio_dias_2025": (
                        cohorte["Dias5_2025"]
                        + cohorte["Dias6_2025"]
                    ).mean(),
                    "promedio_dias_2026": (
                        cohorte["Dias5_2026"]
                        + cohorte["Dias6_2026"]
                    ).mean(),
                    "diferencia_promedio": (
                        (
                            cohorte["Dias5_2026"]
                            + cohorte["Dias6_2026"]
                        ).mean()
                        - (
                            cohorte["Dias5_2025"]
                            + cohorte["Dias6_2025"]
                        ).mean()
                    ),
                }
            )

    return pd.DataFrame(resultados)


def resumir_dimension(
    df: pd.DataFrame,
    dimension: str,
    columna_grupo: str,
    minimo_personas: int = 500,
) -> pd.DataFrame:
    temporal = df.dropna(
        subset=[columna_grupo]
    ).copy()

    resumen = (
        temporal.groupby(
            ["subsistema_2025", columna_grupo],
            as_index=False,
        )
        .agg(
            personas=("ID_persona", "size"),
            promedio_mayo_junio_2025=(
                "mayo_junio_2025",
                "mean",
            ),
            promedio_mayo_junio_2026=(
                "mayo_junio_2026",
                "mean",
            ),
            perdida_total_dias=(
                "perdida_mayo_junio",
                "sum",
            ),
        )
    )

    resumen = resumen[
        resumen["personas"] >= minimo_personas
    ].copy()

    resumen["diferencia_promedio"] = (
        resumen["promedio_mayo_junio_2026"]
        - resumen["promedio_mayo_junio_2025"]
    )

    resumen["variacion_promedio_pct"] = np.where(
        resumen["promedio_mayo_junio_2025"] != 0,
        (
            resumen["promedio_mayo_junio_2026"]
            / resumen["promedio_mayo_junio_2025"]
            - 1
        )
        * 100,
        np.nan,
    )

    perdida_subsistema = resumen.groupby(
        "subsistema_2025"
    )["perdida_total_dias"].transform("sum")

    resumen["contribucion_perdida_pct"] = np.where(
        perdida_subsistema != 0,
        resumen["perdida_total_dias"]
        / perdida_subsistema
        * 100,
        np.nan,
    )

    resumen = resumen.rename(
        columns={
            "subsistema_2025": "subsistema",
            columna_grupo: "grupo",
        }
    )

    resumen.insert(0, "dimension", dimension)

    return resumen


def agregar_dimension_2025(
    panel: pd.DataFrame,
    mapa: pd.DataFrame,
    columna: str,
) -> pd.DataFrame:
    return panel.merge(
        mapa.rename(
            columns={columna: "grupo"}
        ),
        on="ID_persona",
        how="inner",
        validate="one_to_one",
    )


def agregar_transicion(
    panel: pd.DataFrame,
    mapa_2025: pd.DataFrame,
    mapa_2026: pd.DataFrame,
    columna: str,
) -> pd.DataFrame:
    temporal = panel.merge(
        mapa_2025.rename(
            columns={
                columna: f"{columna}_atributo_2025"
            }
        ),
        on="ID_persona",
        how="inner",
        validate="one_to_one",
    )

    temporal = temporal.merge(
        mapa_2026.rename(
            columns={
                columna: f"{columna}_atributo_2026"
            }
        ),
        on="ID_persona",
        how="inner",
        validate="one_to_one",
    )

    temporal["grupo"] = (
        temporal[f"{columna}_atributo_2025"]
        + " → "
        + temporal[f"{columna}_atributo_2026"]
    )

    return temporal


def analizar_centros(
    panel: pd.DataFrame,
    centros_2025: pd.DataFrame,
    centros_2026: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    temporal = panel.merge(
        centros_2025.rename(
            columns={"ID_CENTRO": "centro_2025"}
        ),
        on="ID_persona",
        how="inner",
        validate="one_to_one",
    )

    temporal = temporal.merge(
        centros_2026.rename(
            columns={"ID_CENTRO": "centro_2026"}
        ),
        on="ID_persona",
        how="inner",
        validate="one_to_one",
    )

    # Para atribuir el cambio al centro, se conservan únicamente
    # personas que permanecen en el mismo centro.
    temporal = temporal[
        temporal["centro_2025"]
        == temporal["centro_2026"]
    ].copy()

    centros = (
        temporal.groupby(
            ["subsistema_2025", "centro_2025"],
            as_index=False,
        )
        .agg(
            personas=("ID_persona", "size"),
            promedio_mayo_junio_2025=(
                "mayo_junio_2025",
                "mean",
            ),
            promedio_mayo_junio_2026=(
                "mayo_junio_2026",
                "mean",
            ),
            perdida_total_dias=(
                "perdida_mayo_junio",
                "sum",
            ),
        )
    )

    centros = centros[
        centros["personas"] >= 30
    ].copy()

    centros["diferencia_promedio"] = (
        centros["promedio_mayo_junio_2026"]
        - centros["promedio_mayo_junio_2025"]
    )

    centros["variacion_promedio_pct"] = np.where(
        centros["promedio_mayo_junio_2025"] != 0,
        (
            centros["promedio_mayo_junio_2026"]
            / centros["promedio_mayo_junio_2025"]
            - 1
        )
        * 100,
        np.nan,
    )

    concentracion = []

    for subsistema, grupo in centros.groupby(
        "subsistema_2025"
    ):
        con_perdida = grupo[
            grupo["perdida_total_dias"] > 0
        ].sort_values(
            "perdida_total_dias",
            ascending=False,
        )

        perdida_positiva_total = con_perdida[
            "perdida_total_dias"
        ].sum()

        cantidad_centros = len(con_perdida)
        cantidad_top_10 = max(
            1,
            int(np.ceil(cantidad_centros * 0.10)),
        )

        perdida_top_10 = con_perdida.head(
            cantidad_top_10
        )["perdida_total_dias"].sum()

        perdida_top_20_centros = con_perdida.head(
            20
        )["perdida_total_dias"].sum()

        concentracion.append(
            {
                "subsistema": subsistema,
                "centros_con_perdida": cantidad_centros,
                "centros_top_10_pct": cantidad_top_10,
                "perdida_positiva_total": (
                    perdida_positiva_total
                ),
                "porcentaje_perdida_top_10_pct_centros": (
                    perdida_top_10
                    / perdida_positiva_total
                    * 100
                    if perdida_positiva_total != 0
                    else np.nan
                ),
                "porcentaje_perdida_top_20_centros": (
                    perdida_top_20_centros
                    / perdida_positiva_total
                    * 100
                    if perdida_positiva_total != 0
                    else np.nan
                ),
            }
        )

    centros = centros.rename(
        columns={
            "subsistema_2025": "subsistema",
            "centro_2025": "ID_CENTRO",
        }
    )

    return centros, pd.DataFrame(concentracion)


def main() -> None:
    personas_2025, mapas_2025, ambiguos_2025 = (
        cargar_anio(2025)
    )

    personas_2026, mapas_2026, ambiguos_2026 = (
        cargar_anio(2026)
    )

    panel = personas_2025.merge(
        personas_2026,
        on="ID_persona",
        how="inner",
        suffixes=("_2025", "_2026"),
        validate="one_to_one",
    )

    panel = panel[
        (
            panel["subsistema_2025"]
            == panel["subsistema_2026"]
        )
        & panel["subsistema_2025"].isin(
            ["Secundaria", "UTU"]
        )
    ].copy()

    panel["mayo_junio_2025"] = (
        panel["Dias5_2025"]
        + panel["Dias6_2025"]
    )

    panel["mayo_junio_2026"] = (
        panel["Dias5_2026"]
        + panel["Dias6_2026"]
    )

    panel["perdida_mayo_junio"] = (
        panel["mayo_junio_2025"]
        - panel["mayo_junio_2026"]
    )

    print(
        "\nPersonas del panel estable Secundaria/UTU:",
        f"{len(panel):,}",
    )

    print("\nValores ambiguos por atributo:")
    print("2025:", ambiguos_2025)
    print("2026:", ambiguos_2026)

    # 1. Retención mensual.
    retencion = analizar_retencion(panel)

    retencion.to_csv(
        SALIDA_RETENCION,
        sep=";",
        decimal=",",
        index=False,
    )

    # 2. Dimensiones de interés.
    resultados_focos = []

    dimensiones_base = [
        ("ciclo_2025", "ciclo"),
        ("grado_2025", "grado"),
        ("zona_2025", "ZONA"),
        ("tipo_centro_2025", "tipo_centro"),
        ("departamento_2025", "dept_nombre"),
        ("sexo_2025", "Sexo"),
        ("contexto_2025", "CONTEXTO"),
    ]

    for nombre_dimension, columna in dimensiones_base:
        temporal = agregar_dimension_2025(
            panel,
            mapas_2025[columna],
            columna,
        )

        resultados_focos.append(
            resumir_dimension(
                temporal,
                nombre_dimension,
                "grupo",
            )
        )

    for nombre_dimension, columna in [
        ("transicion_ciclo", "ciclo"),
        ("transicion_grado", "grado"),
    ]:
        temporal = agregar_transicion(
            panel,
            mapas_2025[columna],
            mapas_2026[columna],
            columna,
        )

        resultados_focos.append(
            resumir_dimension(
                temporal,
                nombre_dimension,
                "grupo",
            )
        )

    focos = pd.concat(
        resultados_focos,
        ignore_index=True,
    )

    focos.to_csv(
        SALIDA_FOCOS,
        sep=";",
        decimal=",",
        index=False,
    )

    # 3. Concentración por centros.
    centros, concentracion = analizar_centros(
        panel,
        mapas_2025["ID_CENTRO"],
        mapas_2026["ID_CENTRO"],
    )

    centros.to_csv(
        SALIDA_CENTROS,
        sep=";",
        decimal=",",
        index=False,
    )

    concentracion.to_csv(
        SALIDA_CONCENTRACION,
        sep=";",
        decimal=",",
        index=False,
    )

    print("\n" + "=" * 90)
    print("RETENCIÓN DESDE ABRIL")
    print("=" * 90)

    print(
        retencion.to_string(
            index=False,
            float_format=lambda valor: f"{valor:.2f}",
        )
    )

    print("\n" + "=" * 90)
    print("PRINCIPALES FOCOS DE LA CAÍDA")
    print("=" * 90)

    for subsistema in ["Secundaria", "UTU"]:
        for dimension in focos["dimension"].unique():
            tabla = focos[
                (focos["subsistema"] == subsistema)
                & (focos["dimension"] == dimension)
            ].sort_values(
                "contribucion_perdida_pct",
                ascending=False,
            ).head(5)

            if tabla.empty:
                continue

            print(
                f"\n--- {subsistema} | {dimension} ---"
            )

            print(
                tabla[
                    [
                        "grupo",
                        "personas",
                        "promedio_mayo_junio_2025",
                        "promedio_mayo_junio_2026",
                        "diferencia_promedio",
                        "variacion_promedio_pct",
                        "perdida_total_dias",
                        "contribucion_perdida_pct",
                    ]
                ].to_string(
                    index=False,
                    float_format=lambda valor: (
                        f"{valor:.2f}"
                    ),
                )
            )

    print("\n" + "=" * 90)
    print("CONCENTRACIÓN DE LA CAÍDA POR CENTROS")
    print("=" * 90)

    print(
        concentracion.to_string(
            index=False,
            float_format=lambda valor: f"{valor:.2f}",
        )
    )

    print("\nArchivos generados:")
    print(SALIDA_RETENCION)
    print(SALIDA_FOCOS)
    print(SALIDA_CENTROS)
    print(SALIDA_CONCENTRACION)


if __name__ == "__main__":
    main()