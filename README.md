# Reto-1-IA
📊Análisis de uso de la plataforma CREA (Ceibal) — 2025 vs 2026:
Análisis exploratorio de datos para identificar los factores asociados a la disminución en el uso de la plataforma educativa CREA entre 2025 y 2026, utilizando Python con librerias

Proyecto realizado en el marco de un reto de análisis de datos con bases anonimizadas de Ceibal.

-Contexto y objetivo:
Durante 2026 se observó una caída en la cantidad de usuarios que accedieron a CREA respecto al mismo período de 2025. CREA es la plataforma de gestión del aprendizaje utilizada por los centros educativos públicos de Uruguay.

El objetivo del análisis es explorar los datos disponibles para identificar patrones, generar hipótesis y obtener evidencia que ayude a explicar esta disminución, como insumo para la toma de decisiones.

-Datos:
Se trabajo con 4 bases de datos anonimizadas(Estudiantes 2025, Docentes 2025, Estudiantes 2026 y Docentes 2026)
Cada registro corresponde a una persona/cargo, con información sobre:

Características del usuario (ej. sexo, rol)
Características del centro educativo (subsistema, departamento, contexto socioeconómico)
Características del grupo o clase (grado / ciclo)
Días de acceso a CREA durante abril, mayo y junio
(Las bases de datos no van a estar incluidas en el repositorio por privacidad)

-Preguntas de análisis:
Usuarios y subsistemas:
¿La disminución del uso fue similar entre estudiantes y docentes?
¿Existen diferencias entre subsistemas educativos?
¿Hay departamentos o zonas donde la caída fue mayor?

Contexto y segmentación:
¿Cómo varía el uso según grado o ciclo educativo?
¿Influye el contexto socioeconómico del centro?
¿Existen diferencias según sexo?

-Estructura:
├── scripts/       # scripts de procesamiento y auditoría
├── src/           # código fuente reutilizable
├── data/          # datos (no versionados por privacidad)
├── reports/       # hallazgos y auditorías
└── dashboards/    # visualizaciones

-Autores:
Domingo Leiva 
Santiago Alba
