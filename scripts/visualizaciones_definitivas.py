from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


# -----------------------------------------------------------------------------
# Configuración
# -----------------------------------------------------------------------------

ROOT = Path('.')
OUT = ROOT / 'visualizaciones_ceibal'
OUT.mkdir(parents=True, exist_ok=True)

FILES = {
    'subsistemas': ROOT / 'resultado_subsistemas_panel.csv',
    'meses': ROOT / 'resultado_mes_por_subsistema.csv',
    'retencion': ROOT / 'resultado_retencion_mensual.csv',
    'focos': ROOT / 'resultado_focos_secundaria_utu.csv',
    'doc_resumen': ROOT / 'resultado_docente_estudiante_resumen.csv',
    'doc_cuartiles': ROOT / 'resultado_docente_estudiante_cuartiles.csv',
    'prediccion': ROOT / 'resultado_prediccion_docente_regresiones.csv',
}

ORDER_SUBS = ['Primaria', 'Secundaria', 'UTU']
ORDER_MONTHS = ['Abril', 'Mayo', 'Junio']
ORDER_PERIODS = ['Abril', 'Mayo', 'Junio', 'Mayo-Junio']
ORDER_QUARTILES = [
    'Q1 mayor caída docente',
    'Q2',
    'Q3',
    'Q4 mejor evolución docente',
]

COLORS = {
    '2025': '#94A3B8',
    '2026': '#2563EB',
    'Primaria': '#16A34A',
    'Secundaria': '#7C3AED',
    'UTU': '#EA580C',
    'text': '#0F172A',
    'muted': '#64748B',
    'grid': '#E2E8F0',
}

CYCLE_LABELS = {
    '3er. ciclo ebi': '3.er ciclo EBI',
    '4to. ciclo': '4.º ciclo',
    'bachillerato': 'Bachillerato',
    'ciclo basico': 'Ciclo básico',
    'educacion basica integrada': 'Educación Básica Integrada',
    'bachillerato tecnologico': 'Bachillerato Tecnológico',
    'bachillerato tecnico profesional': 'Bachillerato Técnico Profesional',
    'formacion profesional basica': 'Formación Profesional Básica',
    'bachillerato figari': 'Bachillerato Figari',
}


# -----------------------------------------------------------------------------
# Utilidades
# -----------------------------------------------------------------------------

def read_result(name: str, required: list[str]) -> pd.DataFrame:
    path = FILES[name]
    if not path.exists():
        raise FileNotFoundError(
            f'No se encontró {path}. Ejecutá primero el script que genera ese CSV.'
        )

    df = pd.read_csv(path, sep=';', decimal=',', low_memory=False)
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f'En {path.name} faltan columnas {missing}. '
            f'Disponibles: {df.columns.tolist()}'
        )
    return df


def fmt(value: float, decimals: int = 1) -> str:
    if pd.isna(value):
        return 's/d'
    return f'{value:.{decimals}f}'.replace('.', ',')


def style(fig: go.Figure, title: str, subtitle: str, height: int = 720) -> go.Figure:
    fig.update_layout(
        template='plotly_white',
        title={
            'text': f'<b>{title}</b><br><sup>{subtitle}</sup>',
            'x': 0.02,
            'xanchor': 'left',
            'font': {'size': 27, 'color': COLORS['text']},
        },
        height=height,
        font={'family': 'Arial, Helvetica, sans-serif', 'size': 16},
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin={'l': 90, 'r': 55, 't': 115, 'b': 85},
        legend={
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': 1.02,
            'xanchor': 'right',
            'x': 1,
        },
        hoverlabel={'font_size': 15},
    )
    fig.update_xaxes(showgrid=False, linecolor=COLORS['grid'])
    fig.update_yaxes(gridcolor=COLORS['grid'])
    return fig


def export(fig: go.Figure, stem: str) -> None:
    try:
        fig.write_image(OUT / f'{stem}.png', width=1600, height=900, scale=2)
        fig.write_image(OUT / f'{stem}.svg', width=1600, height=900)
    except Exception as exc:
        print(f'[AVISO] No se pudo exportar {stem} a PNG/SVG: {exc}')
        print('Instalá o actualizá Kaleido: pip install -U kaleido')


def html_fragment(fig: go.Figure, include_js: bool) -> str:
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs='inline' if include_js else False,
        config={
            'displaylogo': False,
            'responsive': True,
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'grafico_ceibal',
                'width': 1600,
                'height': 900,
                'scale': 2,
            },
        },
    )


def get_value(df: pd.DataFrame, filters: dict[str, object], column: str) -> float:
    part = df.copy()
    for key, value in filters.items():
        part = part[part[key] == value]
    if part.empty:
        return np.nan
    return float(pd.to_numeric(part.iloc[0][column], errors='coerce'))


# -----------------------------------------------------------------------------
# Carga de resultados existentes
# -----------------------------------------------------------------------------

subs = read_result(
    'subsistemas',
    [
        'universo', 'subsistema', 'umbral_dias',
        'promedio_dias_2025', 'promedio_dias_2026',
        'variacion_promedio_pct',
    ],
)
months = read_result(
    'meses',
    ['subsistema', 'mes', 'variacion_promedio_pct'],
)
ret = read_result(
    'retencion',
    [
        'subsistema', 'umbral_abril', 'destino',
        'retencion_2025_pct', 'retencion_2026_pct',
        'diferencia_retencion_pp',
    ],
)
foci = read_result(
    'focos',
    [
        'dimension', 'subsistema', 'grupo', 'personas',
        'contribucion_perdida_pct',
    ],
)
doc_summary = read_result(
    'doc_resumen',
    [
        'subsistema', 'periodo', 'pearson_delta_promedio',
        'centros_docente_y_estudiante_bajan_pct',
    ],
)
doc_quartiles = read_result(
    'doc_cuartiles',
    [
        'subsistema', 'periodo', 'cuartil_docente',
        'delta_docente_promedio', 'delta_estudiante_promedio',
        'delta_tasa5_estudiantes_pp',
    ],
)


# -----------------------------------------------------------------------------
# 1A. Subsistemas: promedio 2025 vs 2026
# -----------------------------------------------------------------------------

sub_panel = subs[
    (subs['universo'] == 'panel_estable')
    & (pd.to_numeric(subs['umbral_dias'], errors='coerce') == 1)
].copy()
sub_panel['subsistema'] = pd.Categorical(
    sub_panel['subsistema'], categories=ORDER_SUBS, ordered=True
)
sub_panel = sub_panel.sort_values('subsistema')

fig_subs = go.Figure()
fig_subs.add_bar(
    x=sub_panel['subsistema'],
    y=sub_panel['promedio_dias_2025'],
    name='2025',
    marker_color=COLORS['2025'],
    text=[fmt(value, 1) for value in sub_panel['promedio_dias_2025']],
    textposition='outside',
)
fig_subs.add_bar(
    x=sub_panel['subsistema'],
    y=sub_panel['promedio_dias_2026'],
    name='2026',
    marker_color=COLORS['2026'],
    text=[
        f'{fmt(value, 1)}<br>({fmt(change, 1)}%)'
        for value, change in zip(
            sub_panel['promedio_dias_2026'],
            sub_panel['variacion_promedio_pct'],
        )
    ],
    textposition='outside',
)
fig_subs.update_layout(barmode='group')
fig_subs.update_yaxes(title='Promedio de días, abril-junio')
style(
    fig_subs,
    '1. La caída de intensidad se concentra en educación media',
    'Panel estable: mismas personas en 2025 y 2026 dentro del mismo subsistema',
)
export(fig_subs, '01_subsistemas_promedio')


# -----------------------------------------------------------------------------
# 1B. Ciclos que concentran la pérdida
# -----------------------------------------------------------------------------

cycles = foci[foci['dimension'] == 'ciclo_2025'].copy()
cycles['group_norm'] = cycles['grupo'].astype('string').str.strip().str.lower()
cycles['label'] = cycles['group_norm'].map(CYCLE_LABELS).fillna(cycles['grupo'])

fig_cycles = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=(
        'Secundaria: distribución de los días perdidos',
        'UTU: distribución de los días perdidos',
    ),
    horizontal_spacing=0.19,
)

for col, subsystem in enumerate(['Secundaria', 'UTU'], start=1):
    part = cycles[cycles['subsistema'] == subsystem].sort_values(
        'contribucion_perdida_pct'
    )
    fig_cycles.add_bar(
        x=part['contribucion_perdida_pct'],
        y=part['label'],
        orientation='h',
        marker_color=COLORS[subsystem],
        text=[f'{fmt(value, 1)}%' for value in part['contribucion_perdida_pct']],
        textposition='outside',
        customdata=part['personas'],
        hovertemplate=(
            '%{y}<br>Contribución: %{x:.2f}%'
            '<br>Personas: %{customdata:,}<extra></extra>'
        ),
        showlegend=False,
        row=1,
        col=col,
    )
    fig_cycles.update_xaxes(
        title='Porcentaje de la pérdida total',
        range=[0, max(70, float(part['contribucion_perdida_pct'].max()) * 1.18)],
        row=1,
        col=col,
    )

sec_share = cycles[
    (cycles['subsistema'] == 'Secundaria')
    & cycles['group_norm'].isin(['3er. ciclo ebi', '4to. ciclo'])
]['contribucion_perdida_pct'].sum()
utu_share = cycles[
    (cycles['subsistema'] == 'UTU')
    & cycles['group_norm'].isin(
        ['educacion basica integrada', 'bachillerato tecnologico']
    )
]['contribucion_perdida_pct'].sum()

style(
    fig_cycles,
    'Los ciclos principales concentran la mayor parte de la pérdida',
    (
        f'Secundaria: {fmt(sec_share, 1)}% en 3.er ciclo EBI + 4.º ciclo. '
        f'UTU: {fmt(utu_share, 1)}% en EBI + Bachillerato Tecnológico.'
    ),
    height=760,
)
fig_cycles.update_layout(showlegend=False)
export(fig_cycles, '01b_ciclos_contribucion')


# -----------------------------------------------------------------------------
# 2A. Meses: abril, mayo y junio
# -----------------------------------------------------------------------------

months['mes'] = pd.Categorical(months['mes'], categories=ORDER_MONTHS, ordered=True)
months = months.sort_values(['subsistema', 'mes'])

fig_months = go.Figure()
for subsystem in ORDER_SUBS:
    part = months[months['subsistema'] == subsystem]
    fig_months.add_scatter(
        x=part['mes'],
        y=part['variacion_promedio_pct'],
        mode='lines+markers+text',
        name=subsystem,
        line={'width': 4, 'color': COLORS[subsystem]},
        marker={'size': 12, 'color': COLORS[subsystem]},
        text=[f'{fmt(value, 1)}%' for value in part['variacion_promedio_pct']],
        textposition='top center',
        hovertemplate='%{x}<br>Variación: %{y:.2f}%<extra></extra>',
    )
fig_months.add_hline(y=0, line_dash='dash', line_color=COLORS['muted'])
fig_months.update_xaxes(title='Mes')
fig_months.update_yaxes(title='Variación del promedio de días (%)', ticksuffix='%')
style(
    fig_months,
    '2. El deterioro se profundiza en mayo y junio',
    'Variación 2026 versus 2025 dentro del panel estable de cada subsistema',
)
export(fig_months, '02_meses_por_subsistema')


# -----------------------------------------------------------------------------
# 2B. Continuidad desde abril
# -----------------------------------------------------------------------------

ret5 = ret[
    (pd.to_numeric(ret['umbral_abril'], errors='coerce') == 5)
    & (ret['destino'] == 'Mayo y junio')
    & ret['subsistema'].isin(['Secundaria', 'UTU'])
].copy()
ret5['subsistema'] = pd.Categorical(
    ret5['subsistema'], categories=['Secundaria', 'UTU'], ordered=True
)
ret5 = ret5.sort_values('subsistema')

fig_ret = go.Figure()
fig_ret.add_bar(
    x=ret5['subsistema'],
    y=ret5['retencion_2025_pct'],
    name='2025',
    marker_color=COLORS['2025'],
    text=[f'{fmt(value, 1)}%' for value in ret5['retencion_2025_pct']],
    textposition='outside',
)
fig_ret.add_bar(
    x=ret5['subsistema'],
    y=ret5['retencion_2026_pct'],
    name='2026',
    marker_color=COLORS['2026'],
    text=[
        f'{fmt(value, 1)}%<br>({fmt(diff, 1)} pp)'
        for value, diff in zip(
            ret5['retencion_2026_pct'], ret5['diferencia_retencion_pp']
        )
    ],
    textposition='outside',
)
fig_ret.update_layout(barmode='group')
fig_ret.update_yaxes(title='Retención del uso sostenido (%)', ticksuffix='%', range=[0, 100])
style(
    fig_ret,
    'Quienes comienzan abril con uso frecuente sostienen menos la actividad',
    'Cohorte: estudiantes con al menos 5 días en abril en ambos años',
)
export(fig_ret, '02b_retencion_desde_abril')


# -----------------------------------------------------------------------------
# 3A. Correlación docente-estudiante por período
# -----------------------------------------------------------------------------

doc_summary['periodo'] = pd.Categorical(
    doc_summary['periodo'], categories=ORDER_PERIODS, ordered=True
)
doc_summary = doc_summary.sort_values(['subsistema', 'periodo'])

fig_corr = go.Figure()
for subsystem in ['Secundaria', 'UTU']:
    part = doc_summary[doc_summary['subsistema'] == subsystem]
    fig_corr.add_scatter(
        x=part['periodo'],
        y=part['pearson_delta_promedio'],
        mode='lines+markers+text',
        name=subsystem,
        line={'width': 4, 'color': COLORS[subsystem]},
        marker={'size': 12, 'color': COLORS[subsystem]},
        text=[fmt(value, 2) for value in part['pearson_delta_promedio']],
        textposition='top center',
        hovertemplate='%{x}<br>Pearson: %{y:.3f}<extra></extra>',
    )
fig_corr.update_xaxes(title='Período')
fig_corr.update_yaxes(
    title='Correlación entre cambio docente y estudiantil', range=[0, 0.52]
)
style(
    fig_corr,
    '3. La asociación docente-estudiante se fortalece en mayo y junio',
    'Análisis por centro: cambio promedio docente frente a cambio promedio estudiantil',
)
export(fig_corr, '03_correlacion_docente_estudiante')


# -----------------------------------------------------------------------------
# 3B. Gradiente por cuartiles docentes
# -----------------------------------------------------------------------------

quartiles = doc_quartiles[
    (doc_quartiles['periodo'] == 'Mayo-Junio')
    & doc_quartiles['subsistema'].isin(['Secundaria', 'UTU'])
].copy()
quartiles['cuartil_docente'] = pd.Categorical(
    quartiles['cuartil_docente'], categories=ORDER_QUARTILES, ordered=True
)
quartiles = quartiles.sort_values(['subsistema', 'cuartil_docente'])
short_quartile = {
    'Q1 mayor caída docente': 'Q1<br>Mayor caída',
    'Q2': 'Q2',
    'Q3': 'Q3',
    'Q4 mejor evolución docente': 'Q4<br>Mejor evolución',
}

fig_quartiles = go.Figure()
for subsystem in ['Secundaria', 'UTU']:
    part = quartiles[quartiles['subsistema'] == subsystem]
    fig_quartiles.add_bar(
        x=[short_quartile[str(value)] for value in part['cuartil_docente']],
        y=part['delta_estudiante_promedio'],
        name=subsystem,
        marker_color=COLORS[subsystem],
        text=[fmt(value, 2) for value in part['delta_estudiante_promedio']],
        textposition='outside',
        customdata=np.stack(
            [part['delta_docente_promedio'], part['delta_tasa5_estudiantes_pp']],
            axis=-1,
        ),
        hovertemplate=(
            '%{x}<br>Cambio estudiantil: %{y:.2f} días'
            '<br>Cambio docente: %{customdata[0]:.2f} días'
            '<br>Cambio tasa ≥5 días: %{customdata[1]:.2f} pp'
            '<extra></extra>'
        ),
    )
fig_quartiles.update_layout(barmode='group')
fig_quartiles.update_xaxes(title='Cuartil según evolución docente del centro')
fig_quartiles.update_yaxes(title='Cambio estudiantil en mayo-junio (días)')
style(
    fig_quartiles,
    'Cuanto más cae la actividad docente, más cae la estudiantil',
    'Q1 reúne los centros con mayor caída docente; Q4, los de mejor evolución',
)
export(fig_quartiles, '03b_gradiente_por_cuartiles')


# -----------------------------------------------------------------------------
# 4. Resumen visual final
# -----------------------------------------------------------------------------

sec_change = get_value(sub_panel, {'subsistema': 'Secundaria'}, 'variacion_promedio_pct')
utu_change = get_value(sub_panel, {'subsistema': 'UTU'}, 'variacion_promedio_pct')
sec_ret = get_value(ret5, {'subsistema': 'Secundaria'}, 'diferencia_retencion_pp')
utu_ret = get_value(ret5, {'subsistema': 'UTU'}, 'diferencia_retencion_pp')
sec_corr = get_value(
    doc_summary,
    {'subsistema': 'Secundaria', 'periodo': 'Mayo-Junio'},
    'pearson_delta_promedio',
)
utu_corr = get_value(
    doc_summary,
    {'subsistema': 'UTU', 'periodo': 'Mayo-Junio'},
    'pearson_delta_promedio',
)

limitation = 'La asociación docente-estudiante no demuestra causalidad.'
if FILES['prediccion'].exists():
    pred = pd.read_csv(FILES['prediccion'], sep=';', decimal=',', low_memory=False)
    needed = {'universo_docente', 'subsistema', 'modelo', 'p_valor'}
    if needed.issubset(pred.columns):
        robust = pred[
            (pred['universo_docente'] == 'solo_docentes_un_centro')
            & (pred['modelo'] == '3_ajustado_completo')
        ]
        p_sec = get_value(robust, {'subsistema': 'Secundaria'}, 'p_valor')
        p_utu = get_value(robust, {'subsistema': 'UTU'}, 'p_valor')
        limitation = (
            f'Precedencia temporal no robusta en Secundaria (p={fmt(p_sec, 3)}); '
            f'en UTU la señal fue exploratoria (p={fmt(p_utu, 3)}).'
        )

fig_summary = go.Figure()
cards = [
    {
        'x0': 0.02,
        'x1': 0.32,
        'fill': '#EEF2FF',
        'line': COLORS['Secundaria'],
        'title': '1. DÓNDE',
        'text': (
            f'<b>Secundaria: {fmt(sec_change, 1)}%</b><br>'
            f'<b>UTU: {fmt(utu_change, 1)}%</b><br><br>'
            f'Ciclos principales:<br>{fmt(sec_share, 1)}% y {fmt(utu_share, 1)}% '
            'de la pérdida.'
        ),
    },
    {
        'x0': 0.35,
        'x1': 0.65,
        'fill': '#FFF7ED',
        'line': COLORS['UTU'],
        'title': '2. CUÁNDO',
        'text': (
            '<b>Mayo y junio</b><br><br>'
            'Retención del uso ≥5 días:<br>'
            f'Secundaria {fmt(sec_ret, 1)} pp<br>'
            f'UTU {fmt(utu_ret, 1)} pp'
        ),
    },
    {
        'x0': 0.68,
        'x1': 0.98,
        'fill': '#F0FDF4',
        'line': COLORS['Primaria'],
        'title': '3. RELACIÓN',
        'text': (
            '<b>Actividad docente del centro</b><br><br>'
            'Correlación mayo-junio:<br>'
            f'Secundaria r={fmt(sec_corr, 2)}<br>'
            f'UTU r={fmt(utu_corr, 2)}'
        ),
    },
]

for card in cards:
    fig_summary.add_shape(
        type='rect',
        x0=card['x0'], x1=card['x1'], y0=0.34, y1=0.88,
        xref='paper', yref='paper',
        fillcolor=card['fill'],
        line={'color': card['line'], 'width': 2},
    )
    center = (card['x0'] + card['x1']) / 2
    fig_summary.add_annotation(
        x=center, y=0.80, xref='paper', yref='paper',
        text=f"<b>{card['title']}</b>", showarrow=False,
        font={'size': 22, 'color': card['line']},
    )
    fig_summary.add_annotation(
        x=center, y=0.57, xref='paper', yref='paper',
        text=card['text'], showarrow=False, align='center',
        font={'size': 18, 'color': COLORS['text']},
    )

fig_summary.add_annotation(
    x=0.5, y=0.20, xref='paper', yref='paper', showarrow=False,
    text=(
        '<b>Lectura integrada:</b> la caída de CREA se comporta como una '
        'pérdida de continuidad institucional del uso, concentrada en '
        'educación media y profundizada después de abril.'
    ),
    font={'size': 20, 'color': COLORS['text']},
)
fig_summary.add_annotation(
    x=0.5, y=0.08, xref='paper', yref='paper', showarrow=False,
    text=f'<i>Limitación:</i> {limitation}',
    font={'size': 14, 'color': COLORS['muted']},
)
fig_summary.update_xaxes(visible=False, range=[0, 1])
fig_summary.update_yaxes(visible=False, range=[0, 1])
fig_summary.update_layout(
    template='plotly_white',
    title={'text': '<b>Resumen de los tres hallazgos</b>', 'x': 0.5},
    height=720,
    margin={'l': 35, 'r': 35, 't': 90, 'b': 30},
)
export(fig_summary, '04_resumen_ejecutivo')


# -----------------------------------------------------------------------------
# Dashboard HTML único
# -----------------------------------------------------------------------------

figures = [
    ('Punto 1 — Subsistemas', 'La reducción se concentra en Secundaria y UTU.', fig_subs),
    (
        'Punto 1 — Ciclos',
        'Los porcentajes distribuyen los días perdidos dentro de cada subsistema; no representan causalidad.',
        fig_cycles,
    ),
    ('Punto 2 — Meses', 'El deterioro se profundiza durante mayo y junio.', fig_months),
    (
        'Punto 2 — Continuidad',
        'Los usuarios frecuentes de abril sostienen menos su actividad durante mayo y junio de 2026.',
        fig_ret,
    ),
    (
        'Punto 3 — Relación docente-estudiante',
        'La asociación por centro se fortalece en los meses de mayor deterioro.',
        fig_corr,
    ),
    (
        'Punto 3 — Gradiente institucional',
        'Los centros con mayor caída docente presentan la mayor caída estudiantil.',
        fig_quartiles,
    ),
    ('Resumen ejecutivo', 'Síntesis lista para la diapositiva final.', fig_summary),
]

sections = []
for index, (title, text, figure) in enumerate(figures):
    sections.append(
        f'''<section class="section">
            <div class="head"><h2>{title}</h2><p>{text}</p></div>
            <div class="chart">{html_fragment(figure, include_js=index == 0)}</div>
        </section>'''
    )

page = f'''<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ceibal CREA — Tres hallazgos</title>
<style>
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: #F8FAFC; color: #0F172A; font-family: Arial, sans-serif; }}
.hero {{ padding: 58px 24px 44px; color: white; background: linear-gradient(135deg, #0F172A, #1E3A8A); }}
.wrap {{ width: min(1380px, 96vw); margin: auto; }}
.hero h1 {{ margin: 0 0 12px; font-size: clamp(36px, 5vw, 62px); }}
.hero p {{ margin: 0; max-width: 950px; color: #DBEAFE; font-size: 20px; line-height: 1.5; }}
main {{ padding: 35px 0 70px; }}
.section {{ margin-bottom: 42px; }}
.head {{ padding: 0 8px 14px; }}
.head h2 {{ margin: 0 0 7px; font-size: 28px; }}
.head p {{ margin: 0; color: #64748B; font-size: 17px; }}
.chart {{ overflow: hidden; border: 1px solid #E2E8F0; border-radius: 20px; background: white; box-shadow: 0 12px 35px rgba(15,23,42,.08); }}
.files {{ padding: 24px; border: 1px solid #E2E8F0; border-radius: 18px; background: white; }}
code {{ padding: 2px 7px; border-radius: 7px; background: #E2E8F0; }}
footer {{ padding: 24px; color: #64748B; text-align: center; }}
</style>
</head>
<body>
<header class="hero"><div class="wrap">
<h1>Uso de CREA: tres hallazgos clave</h1>
<p>Comparación abril-mayo-junio de 2025 y 2026 mediante paneles longitudinales de las mismas personas.</p>
</div></header>
<main class="wrap">
{''.join(sections)}
<section class="files"><h2>Archivos para la presentación</h2>
<p>La carpeta <code>{OUT.name}</code> contiene cada gráfico en PNG de alta resolución y SVG editable.</p>
</section>
</main>
<footer>Challenge universitario Ceibal — análisis exploratorio 2025-2026</footer>
</body>
</html>'''

dashboard = OUT / 'dashboard_3_hallazgos.html'
dashboard.write_text(page, encoding='utf-8')

print('\nVisualizaciones generadas correctamente.')
print(f'Dashboard: {dashboard}')
print('Carpeta de imágenes:', OUT)