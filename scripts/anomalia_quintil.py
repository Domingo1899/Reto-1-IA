"""
composicion_quintil.py — ¿La caída del quintil 5 es real o es composición?

EL DEBATE
Nuestro análisis transversal dio que el quintil 5 es el que más cae en uso
sostenido (-9.3 pp urbano). Otro grupo sostiene que eso es un ESPEJISMO DE
COMPOSICIÓN: el Q5 habría recibido la camada de ingresantes más grande, los
ingresantes usan menos CREA, y eso arrastra el promedio hacia abajo aunque los
alumnos que ya estaban usen igual o más que antes.

Las dos cosas pueden ser ciertas a la vez, y el desacuerdo se resuelve con
datos, no con opiniones.

POR QUÉ NO ALCANZA CON COMPARAR "YA ESTABAN" VS "NUEVOS"
Podemos marcar quién es nuevo en 2026 (no aparecía en 2025). NO podemos marcar
quién era nuevo en 2025, porque no tenemos 2024. Entonces comparar el 2025
completo contra un 2026 al que le sacamos los ingresantes es asimétrico: le
quitamos los recién llegados a un año y no al otro. Esa comparación siempre va
a favorecer la tesis de la composición, aunque sea falsa.

LA COMPARACIÓN CORRECTA (panel balanceado)
Quedarse SOLO con las personas presentes en ambos años y comparar su tasa 2025
contra su tasa 2026. Mismas personas, misma base: lo que cambie ahí es
comportamiento puro, sin composición posible.

    Si en el panel el Q5 sigue cayendo fuerte  -> la caída es real.
    Si en el panel el Q5 cae poco o sube       -> era composición.

DESCOMPOSICIÓN
    Δ_total       = tasa2026 - tasa2025        (transversal, lo que reportamos)
    Δ_panel       = tasa2026 - tasa2025 SOLO en los que están en ambos años
    Δ_composicion = Δ_total - Δ_panel          (entradas y salidas de gente)

TERCER CHEQUEO
Ver de qué grado vienen los ingresantes. Si son casi todos de 1º, la diferencia
es de EDAD, no de quintil: los de 1º usan menos CREA en cualquier contexto
socioeconómico. En ese caso la lectura correcta no es "el Q5 usó más que nunca"
sino "el Q5 tuvo proporcionalmente más ingresantes de 1º".

NOTA SOBRE EL QUINTIL
El quintil es un atributo del CENTRO, no de la persona: alguien que cambia de
escuela cambia de quintil. Para el panel se usa el quintil de 2025 (el de
origen) y se reporta cuántos cambiaron, para que se vea el tamaño del problema.
"""

import re
import unicodedata
from pathlib import Path

import pandas as pd

CARPETA = Path("data/interim/entrega-02-corregida")
OUT = Path("data/processed/metricas")

POBLACION = "estudiantes"
MESES = ["Dias4", "Dias5", "Dias6"]
UMBRAL = 10  # uso sostenido: es donde apareció el hallazgo del Q5
EXCL = {"", "sin clasificar"}
CSV_KW = dict(index=False, sep=";", decimal=",")


def norm(v) -> str:
    if pd.isna(v):
        return ""
    s = str(v).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def cargar() -> pd.DataFrame:
    partes = []
    for anio in (2025, 2026):
        df = pd.read_parquet(CARPETA / f"{POBLACION}_{anio}.parquet")
        df.columns = [norm(c).replace(" ", "_") for c in df.columns]
        df = df.drop_duplicates()

        df["ctx"] = df["contexto"].map(norm)
        for c in MESES:
            df[norm(c)] = pd.to_numeric(df[norm(c)], errors="coerce")
        df["dias_total"] = df[[norm(c) for c in MESES]].sum(axis=1, min_count=1)
        df["ok"] = df["dias_total"] >= UMBRAL
        df["anio"] = anio
        partes.append(df)

    d = pd.concat(partes, ignore_index=True)

    # Solo registros con quintil válido (CONTEXTO existe solo en primaria).
    d = d[~d["ctx"].isin(EXCL)].copy()
    d["q"] = d["ctx"].map(lambda v: int(re.search(r"(\d)", v).group()))
    return d


def tasa(df, por):
    g = df.groupby(por, dropna=False).agg(n=("ok", "size"), acc=("ok", "sum")).reset_index()
    g["tasa"] = (100 * g["acc"] / g["n"]).round(1)
    return g


def main():
    pd.set_option("display.width", 200)
    print(f"Cargando {POBLACION} (solo primaria, con quintil)...")
    d = cargar()
    a25 = d[d["anio"] == 2025]
    a26 = d[d["anio"] == 2026]
    print(f"  2025: {len(a25):,} | 2026: {len(a26):,}")

    ids25 = set(a25["id_persona"])
    ids26 = set(a26["id_persona"])
    panel = ids25 & ids26
    print(f"  presentes en ambos años (panel): {len(panel):,}")
    print(f"  ingresantes 2026 (no estaban en 2025): {len(ids26 - ids25):,}")
    print(f"  egresados/salidas (no están en 2026): {len(ids25 - ids26):,}")

    # ================================================================
    print("\n" + "=" * 74)
    print("1. TRANSVERSAL vs PANEL: la comparación que zanja el debate")
    print("=" * 74)

    # transversal: toda la gente de cada año (lo que reportamos originalmente)
    t_cross = tasa(d, ["q", "anio"]).pivot_table(index="q", columns="anio", values="tasa")
    t_cross.columns = [f"cross_{c}" for c in t_cross.columns]
    t_cross["cross_var_pp"] = (t_cross["cross_2026"] - t_cross["cross_2025"]).round(1)

    # panel: SOLO las mismas personas, con el quintil de 2025 fijado
    q_origen = a25.drop_duplicates("id_persona").set_index("id_persona")["q"]
    p = d[d["id_persona"].isin(panel)].copy()
    p["q_origen"] = p["id_persona"].map(q_origen)

    t_panel = tasa(p, ["q_origen", "anio"]).pivot_table(
        index="q_origen", columns="anio", values="tasa")
    t_panel.columns = [f"panel_{c}" for c in t_panel.columns]
    t_panel["panel_var_pp"] = (t_panel["panel_2026"] - t_panel["panel_2025"]).round(1)
    t_panel.index.name = "q"

    comp = t_cross.join(t_panel)
    comp["composicion_pp"] = (comp["cross_var_pp"] - comp["panel_var_pp"]).round(1)
    comp = comp.reset_index()

    print()
    print(comp.to_string(index=False))
    print("""
  CÓMO LEERLO:
    cross_var_pp    -> lo que se ve en el gráfico transversal (nuestro -9.3)
    panel_var_pp    -> las MISMAS personas, 2025 vs 2026. Comportamiento puro.
    composicion_pp  -> la parte del cambio que se explica por entradas/salidas
                       de gente, no por cambio de conducta.
    """)

    q5 = comp[comp["q"] == 5].iloc[0]
    q1 = comp[comp["q"] == 1].iloc[0]
    print(f"  Q5: transversal {q5['cross_var_pp']:+.1f} pp | "
          f"panel {q5['panel_var_pp']:+.1f} pp | composición {q5['composicion_pp']:+.1f} pp")
    print(f"  Q1: transversal {q1['cross_var_pp']:+.1f} pp | "
          f"panel {q1['panel_var_pp']:+.1f} pp | composición {q1['composicion_pp']:+.1f} pp")

    print("\n  VEREDICTO:")
    if q5["panel_var_pp"] > -2:
        print("  En el panel el Q5 casi no cae. La caída transversal del Q5 era")
        print("  efecto de COMPOSICIÓN: tenían razón tus compañeros, y hay que")
        print("  corregir el texto del reporte.")
    elif abs(q5["panel_var_pp"]) > abs(q5["cross_var_pp"]) * 0.7:
        print("  En el panel el Q5 sigue cayendo casi lo mismo. La caída es de")
        print("  COMPORTAMIENTO, no de composición: el hallazgo original se")
        print("  sostiene y la objeción no aplica.")
    else:
        print("  El panel cae, pero bastante menos que el transversal: conviven")
        print("  ambos efectos. Hay que reportar los dos números, no uno solo.")

    if q5["panel_var_pp"] < q1["panel_var_pp"]:
        print("  Además, en el panel el Q5 sigue cayendo MÁS que el Q1.")
    else:
        print("  Además, en el panel el Q5 YA NO cae más que el Q1: el orden")
        print("  entre quintiles se invierte al sacar la composición.")

    # ================================================================
    print("\n" + "=" * 74)
    print("2. ¿QUIÉNES SON LOS INGRESANTES Y CUÁNTO PESAN EN CADA QUINTIL?")
    print("=" * 74)

    a26 = a26.copy()
    a26["grupo"] = a26["id_persona"].isin(panel).map({True: "ya_estaban", False: "nuevos"})
    g = tasa(a26, ["q", "grupo"])
    piv = g.pivot_table(index="q", columns="grupo", values=["n", "tasa"])
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    piv["%_nuevos"] = (100 * piv["n_nuevos"] /
                       (piv["n_nuevos"] + piv["n_ya_estaban"])).round(1)
    piv["brecha_pp"] = (piv["tasa_nuevos"] - piv["tasa_ya_estaban"]).round(1)
    # Arrastre: cuánto baja el promedio del quintil por tener ingresantes.
    piv["arrastre_pp"] = (piv["%_nuevos"] / 100 * piv["brecha_pp"]).round(1)
    piv = piv.reset_index()

    print()
    print(piv.to_string(index=False))
    print("""
  %_nuevos     -> qué proporción del quintil son ingresantes en 2026
  brecha_pp    -> cuánto menos usan los nuevos que los que ya estaban
  arrastre_pp  -> cuánto tira hacia abajo el promedio del quintil ese peso

  La tesis de la composición requiere que %_nuevos sea NOTABLEMENTE mayor en
  el Q5. Si el % de ingresantes es parecido en todos los quintiles, esa
  explicación no alcanza para justificar por qué el Q5 cae más.
    """)

    # ================================================================
    print("\n" + "=" * 74)
    print("3. ¿LOS INGRESANTES SON DE 1º? (¿es efecto edad y no quintil?)")
    print("=" * 74)

    if "grado" in a26.columns:
        nuevos = a26[a26["grupo"] == "nuevos"].copy()
        nuevos["grado_n"] = nuevos["grado"].map(norm)
        dist = (nuevos.groupby("grado_n").size()
                .sort_values(ascending=False).head(10))
        total = len(nuevos)
        print()
        for grado, n in dist.items():
            print(f"  {grado:<28} {n:>7,}  ({100*n/total:>5.1f}%)")
        print("""
  Si la enorme mayoría son de 1º, la brecha de uso de los "nuevos" es un
  efecto de EDAD, no de contexto socioeconómico. En ese caso la conclusión
  correcta no es "el Q5 usó más que nunca" sino "el Q5 recibió
  proporcionalmente más ingresantes de 1º".
        """)
    else:
        print("  No hay columna 'grado'. Paso omitido.")

    # ================================================================
    print("\n" + "=" * 74)
    print("4. CONTROL: ¿cuántos del panel cambiaron de quintil?")
    print("=" * 74)
    q_destino = a26.drop_duplicates("id_persona").set_index("id_persona")["q"]
    cambio = pd.DataFrame({"q25": q_origen, "q26": q_destino}).dropna()
    cambio = cambio[cambio.index.isin(panel)]
    n_cambio = (cambio["q25"] != cambio["q26"]).sum()
    print(f"\n  panel: {len(cambio):,} | cambiaron de quintil: {n_cambio:,} "
          f"({100*n_cambio/len(cambio):.1f}%)")
    print("  (cambiar de quintil = cambiar de centro. Si el % es alto, fijar el")
    print("   quintil de origen introduce algo de ruido y conviene decirlo.)")

    OUT.mkdir(parents=True, exist_ok=True)
    ruta = OUT / "composicion_quintil.csv"
    comp.to_csv(ruta, **CSV_KW)
    print(f"\n  -> guardado {ruta}")


if __name__ == "__main__":
    main()