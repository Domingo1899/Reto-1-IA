"""
chequeo_categorias.py — ¿Qué valores toma cada columna, y son los mismos en 2025 y 2026?

POR QUÉ ESTE CHEQUEO ES EL MÁS IMPORTANTE
El auditor anterior busca errores DENTRO de cada tabla. Este busca errores
ENTRE tablas, que son los que rompen la comparación 2025 vs 2026:

  - Si en 2025 el departamento dice 'Montevideo' y en 2026 'MONTEVIDEO',
    al agrupar te salen 2 departamentos distintos y los conteos se parten.
  - Si en 2025 Rubro tiene 3 categorías y en 2026 tiene 4, la comparación
    por subsistema no es válida sin entender qué pasó.
  - Si aparece una categoría nueva en 2026, puede ser un cambio de
    codificación de Ceibal, y eso EXPLICARÍA parte de la caída aparente.

Este script no juzga: MUESTRA. Vos mirás la salida y decidís qué es un error.
"""

import unicodedata
from pathlib import Path

import pandas as pd

CARPETA = Path("data/interim/entrega-01-original")

# Columnas categóricas que nos interesan (las que usa el PDF para segmentar)
COLUMNAS = [
    "Sexo", "Zona", "ZONA", "tipo_centro", "Rubro", "dept_nombre",
    "ciclo", "grado", "Contexto Sociocultural", "CONTEXTO", "IvsMedia", "IVSMEDIA",
]


def norm(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    s = str(v).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def col_real(df, nombre):
    """Encuentra la columna sin importar mayúsculas: 'Zona' o 'ZONA'."""
    return next((c for c in df.columns if norm(c) == norm(nombre)), None)


def main():
    tablas = {r.stem: pd.read_parquet(r) for r in sorted(CARPETA.glob("*.parquet"))}

    for poblacion in ["docentes", "estudiantes"]:
        a = tablas.get(f"{poblacion}_2025")
        b = tablas.get(f"{poblacion}_2026")
        if a is None or b is None:
            continue

        print("\n" + "#" * 78)
        print(f"#  {poblacion.upper()}")
        print("#" * 78)

        # nombres únicos de columna (Zona/ZONA cuentan como una sola)
        nombres = []
        for c in COLUMNAS:
            n = norm(c)
            if n not in [norm(x) for x in nombres]:
                nombres.append(c)

        for nombre in nombres:
            ca, cb = col_real(a, nombre), col_real(b, nombre)
            if ca is None or cb is None:
                continue

            va = a[ca].dropna().astype(str)
            vb = b[cb].dropna().astype(str)

            # si tiene demasiadas categorías, no es categórica: solo resumimos
            if va.nunique() > 40 or vb.nunique() > 40:
                print(f"\n--- {nombre}: {va.nunique()} / {vb.nunique()} valores únicos "
                      f"(demasiados para listar) ---")
                continue

            print(f"\n{'-' * 78}\n{nombre}   [2025: '{ca}' | 2026: '{cb}']\n{'-' * 78}")

            # Frecuencias de cada valor, año contra año
            fa = va.value_counts()
            fb = vb.value_counts()
            comp = pd.DataFrame({"2025": fa, "2026": fb}).fillna(0).astype(int)
            comp = comp.sort_values("2025", ascending=False)
            print(comp.to_string())

            # ¿Hay variantes de escritura del mismo valor?
            grupos = {}
            for v in set(va.unique()) | set(vb.unique()):
                grupos.setdefault(norm(v), set()).add(v)
            colisiones = {k: sorted(v) for k, v in grupos.items() if len(v) > 1}
            if colisiones:
                print("\n  >>> VARIANTES DE ESCRITURA (son el MISMO valor):")
                for k, v in colisiones.items():
                    print(f"      {v}")

            # ¿Categorías que solo existen en un año?
            solo_25 = {norm(v) for v in va.unique()} - {norm(v) for v in vb.unique()}
            solo_26 = {norm(v) for v in vb.unique()} - {norm(v) for v in va.unique()}
            if solo_25:
                print(f"\n  >>> SOLO EN 2025: {sorted(solo_25)}")
            if solo_26:
                print(f"  >>> SOLO EN 2026: {sorted(solo_26)}")
            if solo_25 or solo_26:
                print("      OJO: si una categoría desaparece o aparece, la comparación")
                print("      entre años NO es directa. Puede ser un cambio de codificación.")


if __name__ == "__main__":
    main()