"""
chequeo_ids.py — ¿Los ID_persona son estables entre 2025 y 2026?

POR QUÉ IMPORTA
Si el ID identifica a la MISMA persona en ambos años, podemos hacer análisis
longitudinal: seguir a cada usuario y ver quién dejó de entrar. Es el hallazgo
más valioso posible para Ceibal.

Si el ID se regenera cada año, ese análisis es IMPOSIBLE, y hacerlo igual
cruzaría personas distintas sin dar ningún error. Un resultado incorrecto en
silencio es peor que un programa que se rompe.

Por eso, antes de asumir nada, lo verificamos.
"""

import re
from pathlib import Path

import pandas as pd

CARPETA = Path("data/raw/entrega-02-corregida")


def identificar(nombre: str):
    n = nombre.lower()
    poblacion = "docentes" if "doc" in n else ("estudiantes" if "estu" in n else None)
    m = re.search(r"20\d{2}", n)
    return poblacion, (int(m.group()) if m else None)


def cargar(ruta: Path) -> pd.DataFrame:
    """Solo las columnas que necesitamos: cargar 35 MB enteros sería un desperdicio."""
    # dtype=str: los ID se leen como TEXTO. Si dejamos que pandas los lea como
    # número, un ID '000123' se convierte en 123 y perdemos los ceros -> los IDs
    # dejan de matchear y creeríamos que no son estables cuando sí lo son.
    df = pd.read_excel(ruta, sheet_name=0, dtype=str)
    return df


def main():
    datos = {}
    for ruta in sorted(CARPETA.glob("*.xlsx")):
        pob, anio = identificar(ruta.stem)
        if pob and anio:
            print(f"Cargando {ruta.name} ...")
            datos[(pob, anio)] = cargar(ruta)

    for poblacion in ["docentes", "estudiantes"]:
        a = datos.get((poblacion, 2025))
        b = datos.get((poblacion, 2026))
        if a is None or b is None:
            print(f"\n!! Falta 2025 o 2026 de {poblacion}, no puedo comparar.")
            continue

        print("\n" + "=" * 70)
        print(f"{poblacion.upper()}")
        print("=" * 70)

        ids_a = a["ID_persona"].dropna()
        ids_b = b["ID_persona"].dropna()

        # --- 1. ¿Cuántos IDs hay, y se repiten dentro del mismo año? ---
        print(f"\n1) VOLUMEN")
        print(f"   2025: {len(ids_a):>8,} filas | {ids_a.nunique():>8,} IDs únicos")
        print(f"   2026: {len(ids_b):>8,} filas | {ids_b.nunique():>8,} IDs únicos")
        if len(ids_a) != ids_a.nunique():
            print(f"   -> En 2025 hay IDs repetidos ({len(ids_a) - ids_a.nunique():,} filas de más).")
            print(f"      Puede ser legítimo: una persona con varios grupos/materias.")

        # --- 2. Formato ---
        print(f"\n2) FORMATO (3 ejemplos de cada año)")
        print(f"   2025: {list(ids_a.head(3))}")
        print(f"   2026: {list(ids_b.head(3))}")
        long_a = ids_a.str.len().value_counts().head(3).to_dict()
        long_b = ids_b.str.len().value_counts().head(3).to_dict()
        print(f"   Largo de los IDs -> 2025: {long_a} | 2026: {long_b}")

        # --- 3. Solapamiento: LA pregunta ---
        set_a, set_b = set(ids_a), set(ids_b)
        comunes = set_a & set_b
        pct = 100 * len(comunes) / len(set_a) if set_a else 0

        print(f"\n3) SOLAPAMIENTO ENTRE AÑOS")
        print(f"   IDs en ambos años: {len(comunes):,}")
        print(f"   = {pct:.1f}% de los IDs de 2025 reaparecen en 2026")

        if pct < 5:
            print("   >>> Los IDs NO son estables: se regeneran cada año.")
            print("       Análisis longitudinal IMPOSIBLE. Solo comparación de grupos.")
        elif pct > 50:
            print("   >>> Los IDs PARECEN estables. Verificamos en el paso 4.")
        else:
            print("   >>> Solapamiento intermedio y sospechoso. Investigar.")

        # --- 4. Test definitivo: ¿un mismo ID describe a la misma persona? ---
        if comunes:
            print(f"\n4) ¿UN MISMO ID ES LA MISMA PERSONA? (test de coherencia)")
            muestra = list(comunes)[:2000]  # muestra, para no tardar

            # Nos quedamos con una fila por ID (la primera) y comparamos Sexo.
            # El sexo de una persona no cambia entre 2025 y 2026.
            sa = a[a["ID_persona"].isin(muestra)].drop_duplicates("ID_persona").set_index("ID_persona")["Sexo"]
            sb = b[b["ID_persona"].isin(muestra)].drop_duplicates("ID_persona").set_index("ID_persona")["Sexo"]
            comp = pd.DataFrame({"sexo_2025": sa, "sexo_2026": sb}).dropna()

            if comp.empty:
                print("   No pude comparar (sin datos suficientes).")
            else:
                iguales = (comp["sexo_2025"] == comp["sexo_2026"]).mean() * 100
                print(f"   Sobre {len(comp):,} IDs comunes, el sexo coincide en el {iguales:.1f}% de los casos.")
                if iguales > 95:
                    print("   >>> CONFIRMADO: el ID identifica a la misma persona entre años.")
                    print("       -> Análisis longitudinal POSIBLE. Es la mejor noticia del proyecto.")
                elif iguales < 70:
                    print("   >>> El ID NO es la misma persona: coincide apenas más que el azar (~50%).")
                    print("       -> Los IDs se REUSAN. Cruzar por ID mezclaría personas distintas.")
                    print("       -> Análisis longitudinal IMPOSIBLE. NO hacer merge por ID.")
                else:
                    print("   >>> Resultado ambiguo. Hay que preguntarle a Ceibal directamente.")


if __name__ == "__main__":
    main()