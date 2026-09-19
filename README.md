# Markowitz + Black-Litterman — Tecnología e Industriales

Aplicación web construida con **Streamlit** para seleccionar acciones de los ETFs sectoriales **XLK (Tecnología)** y **XLI (Industriales)**, analizar cinco años de precios, construir portafolios con **Markowitz** y extender el análisis mediante **Black-Litterman**.

## Funcionalidades

La aplicación está organizada en cuatro fases:

1. **Fase 1 — Obtención y selección de datos**
   - Obtiene holdings actuales de XLK y XLI desde State Street.
   - Descarga cinco años de precios diarios desde Yahoo Finance.
   - Calcula rendimiento acumulado, CAGR, volatilidad y cobertura.
   - Selecciona las acciones con mayor rendimiento histórico por sector.

2. **Fase 2 — Modelo de Markowitz**
   - Rendimientos esperados y matriz de covarianzas anualizada.
   - Matriz de correlaciones.
   - Portafolio de máxima razón de Sharpe.
   - Portafolio de mínima varianza.
   - Equal Weight.
   - Simulación Monte Carlo.
   - Frontera eficiente.
   - Comparación Tecnología vs. Industriales vs. universo combinado.

3. **Fase 3 — Interpretación detallada**
   - Resume los principales resultados de Markowitz.
   - Ayuda a interpretar rendimiento, volatilidad, Sharpe y pesos.
   - Permite regresar a la fase anterior o continuar a Black-Litterman.

4. **Fase 4 — Black-Litterman**
   - Utiliza capitalización bursátil para obtener pesos de mercado.
   - Calcula automáticamente el coeficiente de aversión al riesgo.
   - Tasa libre de riesgo automática y editable.
   - `tau = 0.05` por defecto y editable.
   - Permite una opinión relativa entre dos activos.
   - La opinión se expresa en puntos porcentuales anuales y con confianza de 1% a 100%.
   - Convierte automáticamente la confianza en la matriz de incertidumbre `Omega`.
   - Calcula rendimientos implícitos de equilibrio `pi`.
   - Calcula rendimientos posteriores Black-Litterman.
   - Optimiza un portafolio Black-Litterman y lo compara con Markowitz.
   - Incluye tabla de pesos, cambios e interpretación automática.

## Requisitos

- Python 3.10 o superior recomendado.
- Conexión a Internet para consultar State Street y Yahoo Finance.

## Instalación local

### 1. Clonar el repositorio

```bash
git clone https://github.com/TU-USUARIO/TU-REPOSITORIO.git
cd TU-REPOSITORIO
```

### 2. Crear un entorno virtual

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Windows CMD:

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Ejecutar la aplicación

```bash
streamlit run app.py
```

Streamlit mostrará una dirección local, normalmente:

```text
http://localhost:8501
```

## Subirlo a un repositorio nuevo de GitHub

Desde la carpeta del proyecto:

```bash
git init
git add .
git commit -m "Initial Markowitz and Black-Litterman app"
git branch -M main
git remote add origin https://github.com/TU-USUARIO/TU-REPOSITORIO.git
git push -u origin main
```

También puedes crear primero el repositorio vacío en GitHub y seguir las instrucciones que GitHub muestra para **push an existing repository from the command line**.

## Despliegue en Streamlit Community Cloud

1. Sube este proyecto a GitHub.
2. En Streamlit Community Cloud selecciona **Create app**.
3. Elige el repositorio y la rama `main`.
4. Archivo principal: `app.py`.
5. Despliega la aplicación.

No se requieren claves API privadas para la versión actual.

## Fuentes de datos

- **State Street Global Advisors**: holdings de XLK y XLI.
- **Yahoo Finance**, consultado mediante `yfinance`: precios históricos, capitalización bursátil y referencias de mercado utilizadas por la aplicación.

La disponibilidad y estructura de datos de fuentes externas puede cambiar con el tiempo. El programa incluye manejo de errores, pero puede requerir ajustes si State Street o Yahoo Finance modifican sus interfaces.

## Consideraciones metodológicas

La selección inicial de acciones se realiza utilizando rendimiento histórico de cinco años. Esto introduce **sesgo retrospectivo** y potencial **sesgo de supervivencia**.

Los rendimientos históricos, los resultados de Markowitz y los rendimientos posteriores Black-Litterman **no constituyen predicciones garantizadas ni recomendaciones de inversión**.

Black-Litterman incorpora una opinión del usuario al equilibrio de mercado. El resultado depende de:

- Pesos de mercado.
- Matriz de covarianzas.
- Tasa libre de riesgo.
- Aversión al riesgo.
- Valor de `tau`.
- Opinión relativa introducida.
- Nivel de confianza asignado.

## Estructura del repositorio

```text
.
├── app.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Licencia

Este repositorio no incluye una licencia de uso explícita. Si deseas publicarlo como software abierto, puedes añadir una licencia como MIT, Apache-2.0 o la que corresponda a tu proyecto.

---

## Versión actual

**v2**

### Cambios principales de v2

- Black-Litterman ya no se detiene si Yahoo Finance no entrega capitalización bursátil para todos los activos.
- Jerarquía de pesos de equilibrio:
  1. Capitalización bursátil obtenida de Yahoo Finance.
  2. Capitalización reconstruida con acciones en circulación × último precio.
  3. Benchmark alternativo con pesos oficiales de XLK/XLI obtenidos en la Fase 1.
- La aplicación nunca mezcla capitalizaciones y pesos ETF dentro del mismo cálculo.
- Se muestra explícitamente la fuente de pesos usada en Black-Litterman.
- Se agregó un diagnóstico desplegable de capitalizaciones.
- Se conserva la distribución sectorial de Markowitz y la comparación sectorial con Black-Litterman.
- Se mejoró la redacción de la opinión relativa.
