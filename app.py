import io
import time
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================
st.set_page_config(
    page_title="Markowitz — Tecnología + Industriales",
    page_icon="📈",
    layout="wide",
)

TRADING_DAYS = 252
MIN_COVERAGE = 0.90

STATE_STREET_FILES = {
    "Tecnología": {
        "fund": "XLK",
        "url": "https://www.ssga.com/library-content/products/fund-data/etfs/us/holdings-daily-us-en-xlk.xlsx",
    },
    "Industriales": {
        "fund": "XLI",
        "url": "https://www.ssga.com/library-content/products/fund-data/etfs/us/holdings-daily-us-en-xli.xlsx",
    },
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/128.0 Safari/537.36"
    ),
    "Accept": (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,"
        "application/octet-stream,*/*"
    ),
    "Referer": "https://www.ssga.com/",
}

# ============================================================
# ESTADO DE LA APLICACIÓN
# ============================================================
if "fase" not in st.session_state:
    st.session_state.fase = 1

if "universo_final" not in st.session_state:
    st.session_state.universo_final = None

if "precios_fase1" not in st.session_state:
    st.session_state.precios_fase1 = None

if "fase1_completa" not in st.session_state:
    st.session_state.fase1_completa = False

if "ranking_fase1" not in st.session_state:
    st.session_state.ranking_fase1 = None

if "markowitz_resultados" not in st.session_state:
    st.session_state.markowitz_resultados = None

if "risk_free_markowitz" not in st.session_state:
    st.session_state.risk_free_markowitz = 0.04


# ============================================================
# FUNCIONES — STATE STREET
# ============================================================
def _find_header_row(raw: pd.DataFrame):
    for idx in range(min(len(raw), 40)):
        vals = [
            str(v).strip().lower()
            for v in raw.iloc[idx].tolist()
            if pd.notna(v)
        ]
        joined = " | ".join(vals)

        has_ticker = ("ticker" in joined) or ("symbol" in joined)
        has_name = ("name" in joined) or ("security" in joined)

        if has_ticker and has_name:
            return idx

    raise ValueError(
        "No se pudo localizar la fila de encabezados del archivo de holdings."
    )


def _normalize_column_name(value):
    return str(value).strip().lower().replace("\n", " ")


def _choose_column(columns, candidates):
    normalized = {
        col: _normalize_column_name(col)
        for col in columns
    }

    for candidate in candidates:
        for original, norm in normalized.items():
            if candidate == norm or candidate in norm:
                return original

    return None


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def get_state_street_holdings(sector_label: str) -> pd.DataFrame:
    cfg = STATE_STREET_FILES[sector_label]
    url = cfg["url"]
    fund = cfg["fund"]

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=30,
        allow_redirects=True,
    )
    response.raise_for_status()

    if len(response.content) < 2000:
        raise ValueError(
            f"State Street devolvió un archivo demasiado pequeño para {fund}."
        )

    raw = pd.read_excel(
        io.BytesIO(response.content),
        header=None,
        engine="openpyxl",
    )

    header_row = _find_header_row(raw)

    table = pd.read_excel(
        io.BytesIO(response.content),
        header=header_row,
        engine="openpyxl",
    )

    ticker_col = _choose_column(
        table.columns,
        ["ticker", "symbol"],
    )

    name_col = _choose_column(
        table.columns,
        ["security name", "name", "security"],
    )

    weight_col = _choose_column(
        table.columns,
        ["weight (%)", "weight %", "weight"],
    )

    if ticker_col is None:
        raise ValueError(
            f"No se encontró Ticker/Symbol en el archivo oficial de {fund}."
        )

    out = pd.DataFrame()
    out["Ticker"] = table[ticker_col].astype(str).str.strip()

    if name_col is not None:
        out["Nombre"] = table[name_col].astype(str).str.strip()
    else:
        out["Nombre"] = ""

    if weight_col is not None:
        out["Peso fondo"] = pd.to_numeric(
            table[weight_col],
            errors="coerce",
        )
    else:
        out["Peso fondo"] = np.nan

    out["Sector"] = sector_label
    out["Fondo"] = fund

    out["Ticker"] = (
        out["Ticker"]
        .str.replace(".", "-", regex=False)
        .str.upper()
    )

    invalid = {"", "NAN", "NONE", "CASH", "USD", "-"}

    out = out[
        ~out["Ticker"].isin(invalid)
        & out["Ticker"].str.match(r"^[A-Z0-9\-]+$", na=False)
    ].copy()

    out = out.drop_duplicates("Ticker").reset_index(drop=True)

    if len(out) < 10:
        raise ValueError(
            f"Solo se detectaron {len(out)} holdings válidos para {fund}."
        )

    return out


# ============================================================
# FUNCIONES — YAHOO FINANCE
# ============================================================
def extract_close(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            return pd.DataFrame()
        close = raw["Close"].copy()

    else:
        if "Close" not in raw.columns:
            return pd.DataFrame()

        close = raw[["Close"]].copy()

        if len(tickers) == 1:
            close.columns = tickers

    if isinstance(close, pd.Series):
        close = close.to_frame()

    return close.sort_index()


def _download_batch(batch):
    raw = yf.download(
        tickers=batch,
        period="5y",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="column",
        timeout=25,
    )

    return extract_close(raw, batch)


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def download_prices(tickers_tuple: tuple[str, ...]) -> pd.DataFrame:
    tickers = list(dict.fromkeys(tickers_tuple))

    if not tickers:
        return pd.DataFrame()

    frames = []
    batch_size = 35

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]

        try:
            close = _download_batch(batch)
            if not close.empty:
                frames.append(close)

        except Exception:
            midpoint = max(len(batch) // 2, 1)

            for small in (batch[:midpoint], batch[midpoint:]):
                if not small:
                    continue

                try:
                    close = _download_batch(small)
                    if not close.empty:
                        frames.append(close)
                except Exception:
                    pass

        time.sleep(0.15)

    if not frames:
        return pd.DataFrame()

    prices = pd.concat(frames, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]

    return prices.sort_index()


# ============================================================
# FUNCIONES — FASE 1
# ============================================================
def calculate_ranking(
    prices: pd.DataFrame,
    metadata: pd.DataFrame,
    min_coverage: float = 0.90,
) -> pd.DataFrame:

    available = [
        t for t in metadata["Ticker"].tolist()
        if t in prices.columns
    ]

    if not available:
        return pd.DataFrame()

    sector_prices = prices[available]
    max_obs = int(sector_prices.notna().sum().max())

    rows = []

    for ticker in available:
        s = sector_prices[ticker].dropna()

        if len(s) < 2:
            continue

        coverage = len(s) / max_obs if max_obs else 0

        if coverage < min_coverage:
            continue

        initial = float(s.iloc[0])
        final = float(s.iloc[-1])

        if initial <= 0:
            continue

        total_return = final / initial - 1.0

        days = max(
            (s.index[-1] - s.index[0]).days,
            1,
        )

        years = days / 365.25

        cagr = (final / initial) ** (1 / years) - 1

        daily = s.pct_change(fill_method=None).dropna()

        vol = (
            daily.std() * np.sqrt(TRADING_DAYS)
            if len(daily) > 1
            else np.nan
        )

        rows.append(
            {
                "Ticker": ticker,
                "Precio inicial": initial,
                "Precio final": final,
                "Rendimiento 5Y": total_return,
                "CAGR": cagr,
                "Volatilidad anual": vol,
                "Cobertura": coverage,
                "Fecha inicial": s.index[0].date(),
                "Fecha final": s.index[-1].date(),
            }
        )

    ranking = pd.DataFrame(rows)

    if ranking.empty:
        return ranking

    ranking = ranking.merge(
        metadata[
            ["Ticker", "Nombre", "Sector", "Fondo", "Peso fondo"]
        ],
        on="Ticker",
        how="left",
    )

    return (
        ranking
        .sort_values("Rendimiento 5Y", ascending=False)
        .reset_index(drop=True)
    )


def show_ranking(df: pd.DataFrame):
    show = df.copy()

    show.insert(
        0,
        "Posición",
        range(1, len(show) + 1),
    )

    for col in [
        "Rendimiento 5Y",
        "CAGR",
        "Volatilidad anual",
        "Cobertura",
    ]:
        show[col] *= 100

    st.dataframe(
        show[
            [
                "Posición",
                "Ticker",
                "Nombre",
                "Rendimiento 5Y",
                "CAGR",
                "Volatilidad anual",
                "Cobertura",
            ]
        ],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rendimiento 5Y":
                st.column_config.NumberColumn(format="%.2f %%"),
            "CAGR":
                st.column_config.NumberColumn(format="%.2f %%"),
            "Volatilidad anual":
                st.column_config.NumberColumn(format="%.2f %%"),
            "Cobertura":
                st.column_config.NumberColumn(format="%.1f %%"),
        },
    )


# ============================================================
# FUNCIONES — MARKOWITZ
# ============================================================
def annual_statistics(prices):
    returns = prices.pct_change(fill_method=None).dropna(how="all")
    returns = returns.dropna(axis=0, how="any")

    mean_returns = returns.mean() * TRADING_DAYS
    cov_matrix = returns.cov() * TRADING_DAYS
    corr_matrix = returns.corr()

    return returns, mean_returns, cov_matrix, corr_matrix


def portfolio_performance(
    weights,
    mean_returns,
    cov_matrix,
    risk_free,
):
    weights = np.asarray(weights)

    ret = float(
        np.dot(weights, mean_returns.values)
    )

    vol = float(
        np.sqrt(
            weights.T
            @ cov_matrix.values
            @ weights
        )
    )

    sharpe = (
        (ret - risk_free) / vol
        if vol > 0
        else np.nan
    )

    return ret, vol, sharpe


def min_variance_portfolio(
    mean_returns,
    cov_matrix,
    risk_free,
):
    n = len(mean_returns)
    x0 = np.repeat(1 / n, n)

    bounds = tuple(
        (0.0, 1.0)
        for _ in range(n)
    )

    constraints = ({
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1.0,
    },)

    result = minimize(
        lambda w: portfolio_performance(
            w,
            mean_returns,
            cov_matrix,
            risk_free,
        )[1],
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 2000,
            "ftol": 1e-12,
        },
    )

    if not result.success:
        raise RuntimeError(
            "No se pudo calcular el portafolio de mínima varianza."
        )

    return result.x


def max_sharpe_portfolio(
    mean_returns,
    cov_matrix,
    risk_free,
):
    n = len(mean_returns)
    x0 = np.repeat(1 / n, n)

    bounds = tuple(
        (0.0, 1.0)
        for _ in range(n)
    )

    constraints = ({
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1.0,
    },)

    def negative_sharpe(w):
        _, _, sharpe = portfolio_performance(
            w,
            mean_returns,
            cov_matrix,
            risk_free,
        )

        return -sharpe if np.isfinite(sharpe) else 1e9

    result = minimize(
        negative_sharpe,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 2000,
            "ftol": 1e-12,
        },
    )

    if not result.success:
        raise RuntimeError(
            "No se pudo calcular el portafolio de máximo Sharpe."
        )

    return result.x


def monte_carlo(
    mean_returns,
    cov_matrix,
    risk_free,
    simulations,
):
    rng = np.random.default_rng(42)
    n = len(mean_returns)

    weights = rng.dirichlet(
        np.ones(n),
        size=simulations,
    )

    portfolio_returns = (
        weights @ mean_returns.values
    )

    variances = np.einsum(
        "ij,jk,ik->i",
        weights,
        cov_matrix.values,
        weights,
    )

    vol = np.sqrt(variances)

    sharpe = np.divide(
        portfolio_returns - risk_free,
        vol,
        out=np.full_like(
            portfolio_returns,
            np.nan,
        ),
        where=vol > 0,
    )

    return pd.DataFrame({
        "Rendimiento": portfolio_returns,
        "Volatilidad": vol,
        "Sharpe": sharpe,
    })


def efficient_frontier(
    mean_returns,
    cov_matrix,
    risk_free,
    points=45,
):
    n = len(mean_returns)
    x0 = np.repeat(1 / n, n)

    min_w = min_variance_portfolio(
        mean_returns,
        cov_matrix,
        risk_free,
    )

    min_ret, _, _ = portfolio_performance(
        min_w,
        mean_returns,
        cov_matrix,
        risk_free,
    )

    max_ret = float(mean_returns.max())

    targets = np.linspace(
        min_ret,
        max_ret,
        points,
    )

    rows = []

    for target in targets:
        constraints = (
            {
                "type": "eq",
                "fun": lambda w: np.sum(w) - 1.0,
            },
            {
                "type": "eq",
                "fun": lambda w, t=target:
                    np.dot(
                        w,
                        mean_returns.values,
                    ) - t,
            },
        )

        bounds = tuple(
            (0.0, 1.0)
            for _ in range(n)
        )

        result = minimize(
            lambda w: portfolio_performance(
                w,
                mean_returns,
                cov_matrix,
                risk_free,
            )[1],
            x0=x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={
                "maxiter": 1500,
                "ftol": 1e-10,
            },
        )

        if result.success:
            r, v, s = portfolio_performance(
                result.x,
                mean_returns,
                cov_matrix,
                risk_free,
            )

            rows.append({
                "Rendimiento": r,
                "Volatilidad": v,
                "Sharpe": s,
            })

    return pd.DataFrame(rows)


def weights_table(weights, universe):
    df = universe[
        ["Ticker", "Nombre", "Sector"]
    ].copy()

    df["Peso"] = np.asarray(weights)

    # Mostrar solamente pesos de al menos 0.05%.
    df = df[
        df["Peso"] >= 0.0005
    ].copy()

    return (
        df
        .sort_values("Peso", ascending=False)
        .reset_index(drop=True)
    )


def analyze_subset(
    tickers,
    prices,
    risk_free,
):
    subset = prices[tickers].dropna(
        axis=0,
        how="any",
    )

    _, mu, cov, _ = annual_statistics(subset)

    w_min = min_variance_portfolio(
        mu,
        cov,
        risk_free,
    )

    w_sharpe = max_sharpe_portfolio(
        mu,
        cov,
        risk_free,
    )

    return {
        "min": portfolio_performance(
            w_min,
            mu,
            cov,
            risk_free,
        ),
        "sharpe": portfolio_performance(
            w_sharpe,
            mu,
            cov,
            risk_free,
        ),
    }


# ============================================================
# FUNCIONES — BLACK-LITTERMAN
# ============================================================
@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def get_market_caps(tickers_tuple: tuple[str, ...]) -> pd.Series:
    """Obtiene capitalización bursátil desde Yahoo Finance."""
    caps = {}

    for ticker in tickers_tuple:
        cap = np.nan
        try:
            tk = yf.Ticker(ticker)
            try:
                cap = tk.fast_info.get("market_cap", np.nan)
            except Exception:
                cap = np.nan

            if cap is None or not np.isfinite(float(cap)) or float(cap) <= 0:
                try:
                    cap = tk.info.get("marketCap", np.nan)
                except Exception:
                    cap = np.nan
        except Exception:
            cap = np.nan

        try:
            cap = float(cap)
        except Exception:
            cap = np.nan

        caps[ticker] = cap
        time.sleep(0.03)

    return pd.Series(caps, dtype="float64")


@st.cache_data(ttl=60 * 60, show_spinner=False)
def get_auto_risk_free_rate() -> float:
    """Usa ^IRX como aproximación automática a Treasury de corto plazo."""
    try:
        raw = yf.download(
            "^IRX",
            period="10d",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
            timeout=20,
        )

        if raw is None or raw.empty:
            raise ValueError("Sin datos de ^IRX")

        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"].iloc[:, 0].dropna()
        else:
            close = raw["Close"].dropna()

        if close.empty:
            raise ValueError("Sin cierres de ^IRX")

        return float(close.iloc[-1]) / 100.0
    except Exception:
        return 0.04


def black_litterman_relative_view(
    cov_matrix: pd.DataFrame,
    market_weights: pd.Series,
    risk_free: float,
    tau: float,
    outperformer: str,
    underperformer: str,
    spread: float,
    confidence: float,
):
    """
    Black-Litterman con una sola opinión relativa:
    outperformer - underperformer = spread.

    La confianza se transforma a Omega con una variante intuitiva
    del enfoque de Idzorek:
        Omega = ((1-c)/c) * P (tau*Sigma) P'
    """
    tickers = list(cov_matrix.index)
    sigma = cov_matrix.loc[tickers, tickers].values
    w_mkt = market_weights.reindex(tickers).values.astype(float)

    market_return_proxy = float(np.dot(w_mkt, np.diag(sigma) * 0 + 1.0))
    # El proxy anterior solo inicializa forma; delta se calcula fuera con
    # rendimiento histórico del portafolio de mercado y se inyecta después.

    p = np.zeros((1, len(tickers)))
    p[0, tickers.index(outperformer)] = 1.0
    p[0, tickers.index(underperformer)] = -1.0
    q = np.array([float(spread)])

    c = float(np.clip(confidence, 0.01, 0.999))
    view_variance = float(p @ (tau * sigma) @ p.T)
    omega_value = ((1.0 - c) / c) * view_variance
    omega_value = max(omega_value, 1e-12)
    omega = np.array([[omega_value]])

    return p, q, omega


def black_litterman_posterior(
    cov_matrix: pd.DataFrame,
    pi_excess: pd.Series,
    tau: float,
    p: np.ndarray,
    q: np.ndarray,
    omega: np.ndarray,
) -> pd.Series:
    tickers = list(cov_matrix.index)
    sigma = cov_matrix.loc[tickers, tickers].values
    pi = pi_excess.reindex(tickers).values

    tau_sigma = tau * sigma
    middle = np.linalg.inv(p @ tau_sigma @ p.T + omega)
    posterior = pi + tau_sigma @ p.T @ middle @ (q - p @ pi)

    return pd.Series(posterior, index=tickers, name="BL excess")


# ============================================================
# ENCABEZADO
# ============================================================
st.title(
    "📈 Modelo de Markowitz — Tecnología + Industriales"
)

st.caption(
    "Proceso completo: selección de datos → construcción del portafolio."
)

# Indicador visual de fase
p1, p2, p3, p4 = st.columns(4)

labels = [
    "① Fase 1 — Selección",
    "② Fase 2 — Markowitz",
    "③ Fase 3 — Interpretación",
    "④ Fase 4 — Black-Litterman",
]

for i, col in enumerate([p1, p2, p3, p4], start=1):
    if st.session_state.fase == i:
        col.success(labels[i - 1])
    elif st.session_state.fase > i:
        col.info(labels[i - 1] + " ✓")
    else:
        col.info(labels[i - 1])


# ============================================================
# FASE 1
# ============================================================
if st.session_state.fase == 1:

    st.header(
        "Fase 1 — Obtención y selección de datos"
    )

    st.write(
        "Primero se obtienen los componentes actuales de "
        "**XLK (Tecnología)** y **XLI (Industriales)**. "
        "Posteriormente Yahoo Finance proporciona los precios "
        "diarios de los últimos cinco años."
    )

    st.sidebar.header("Fase 1")

    top_n = st.sidebar.slider(
        "Acciones por sector",
        min_value=5,
        max_value=20,
        value=10,
        step=1,
        disabled=st.session_state.fase1_completa,
    )

    st.sidebar.write("Periodo: **5 años**")
    st.sidebar.write("Cobertura mínima: **90%**")

    # --------------------------------------------------------
    # SI TODAVÍA NO SE HA EJECUTADO LA FASE 1
    # --------------------------------------------------------
    if not st.session_state.fase1_completa:

        st.info(
            "Presiona el botón para comenzar la Fase 1."
        )

        st.markdown(
            """
            **Procedimiento de esta fase**

            1. Obtener holdings oficiales de XLK y XLI.
            2. Descargar precios históricos desde Yahoo Finance.
            3. Calcular rendimiento acumulado de 5 años.
            4. Calcular CAGR y volatilidad anual.
            5. Excluir acciones sin suficiente historial.
            6. Seleccionar las acciones con mayor rendimiento de cada sector.
            """
        )

        if st.button(
            "🔎 Obtener datos y seleccionar acciones",
            type="primary",
            use_container_width=True,
        ):

            universes = {}

            for sector in [
                "Tecnología",
                "Industriales",
            ]:
                fund = STATE_STREET_FILES[
                    sector
                ]["fund"]

                with st.spinner(
                    f"Obteniendo holdings de {fund}..."
                ):
                    universes[sector] = (
                        get_state_street_holdings(
                            sector
                        )
                    )

                st.success(
                    f"{fund}: "
                    f"{len(universes[sector])} "
                    "acciones detectadas."
                )

            all_metadata = pd.concat(
                list(universes.values()),
                ignore_index=True,
            )

            all_tickers = tuple(
                all_metadata["Ticker"]
                .drop_duplicates()
                .tolist()
            )

            with st.spinner(
                "Descargando cinco años de precios "
                "desde Yahoo Finance..."
            ):
                prices = download_prices(
                    all_tickers
                )

            if prices.empty:
                st.error(
                    "Yahoo Finance no devolvió precios."
                )
                st.stop()

            selected = []
            rankings = {}

            for sector in [
                "Tecnología",
                "Industriales",
            ]:
                ranking = calculate_ranking(
                    prices,
                    universes[sector],
                    MIN_COVERAGE,
                )

                if ranking.empty:
                    st.error(
                        f"No se pudo construir "
                        f"el ranking de {sector}."
                    )
                    st.stop()

                top = ranking.head(
                    top_n
                ).copy()

                selected.append(top)
                rankings[sector] = top.copy()

            combined = pd.concat(
                selected,
                ignore_index=True,
            )

            # Guardar TODO en session_state antes del rerun.
            st.session_state.universo_final = (
                combined[
                    [
                        "Ticker",
                        "Nombre",
                        "Sector",
                        "Fondo",
                    ]
                ].copy()
            )

            st.session_state.precios_fase1 = (
                prices.copy()
            )

            st.session_state.ranking_fase1 = (
                rankings
            )

            st.session_state.fase1_completa = True

            st.rerun()

    # --------------------------------------------------------
    # SI YA SE COMPLETÓ LA FASE 1, MOSTRAR RESULTADOS GUARDADOS
    # --------------------------------------------------------
    else:

        rankings = st.session_state.ranking_fase1
        universe_saved = st.session_state.universo_final

        st.success(
            "✅ Fase 1 completada. "
            "Los resultados quedaron guardados."
        )

        for sector in [
            "Tecnología",
            "Industriales",
        ]:
            top = rankings[sector].copy()

            st.divider()
            st.subheader(
                f"🏆 {sector} — Top {len(top)}"
            )

            show_ranking(top)

            graph = top.sort_values(
                "Rendimiento 5Y",
                ascending=True,
            ).copy()

            graph["Rendimiento (%)"] = (
                graph["Rendimiento 5Y"] * 100
            )

            fig = px.bar(
                graph,
                x="Rendimiento (%)",
                y="Ticker",
                orientation="h",
                hover_name="Nombre",
                title=(
                    "Rendimiento acumulado "
                    f"de 5 años — {sector}"
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        st.divider()
        st.subheader(
            "Universo seleccionado para Markowitz"
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Total",
            len(universe_saved),
        )

        c2.metric(
            "Tecnología",
            (
                universe_saved["Sector"]
                == "Tecnología"
            ).sum(),
        )

        c3.metric(
            "Industriales",
            (
                universe_saved["Sector"]
                == "Industriales"
            ).sum(),
        )

        st.dataframe(
            universe_saved,
            use_container_width=True,
            hide_index=True,
        )

        col_next, col_reset = st.columns([3, 1])

        with col_next:
            if st.button(
                "Siguiente → Aplicar Markowitz",
                type="primary",
                use_container_width=True,
            ):
                st.session_state.fase = 2
                st.rerun()

        with col_reset:
            if st.button(
                "Reiniciar Fase 1",
                use_container_width=True,
            ):
                st.session_state.fase1_completa = False
                st.session_state.universo_final = None
                st.session_state.precios_fase1 = None
                st.session_state.ranking_fase1 = None
                st.rerun()


# ============================================================
# FASE 2
# ============================================================
elif st.session_state.fase == 2:

    universe = st.session_state.universo_final

    if universe is None:
        st.warning(
            "Primero es necesario completar "
            "la Fase 1."
        )

        if st.button(
            "← Ir a Fase 1"
        ):
            st.session_state.fase = 1
            st.rerun()

        st.stop()

    st.header(
        "Fase 2 — Modelo de Markowitz"
    )

    st.write(
        "Ahora se utilizan exclusivamente las acciones "
        "seleccionadas en la Fase 1."
    )

    st.sidebar.header("Fase 2")

    risk_free_pct = st.sidebar.number_input(
        "Tasa libre de riesgo anual (%)",
        min_value=0.0,
        max_value=20.0,
        value=4.0,
        step=0.25,
    )

    simulations = st.sidebar.select_slider(
        "Portafolios simulados",
        options=[
            10000,
            25000,
            50000,
            100000,
        ],
        value=50000,
    )

    risk_free = risk_free_pct / 100

    if st.button(
        "← Volver a Fase 1"
    ):
        st.session_state.fase = 1
        st.rerun()

    tickers = universe[
        "Ticker"
    ].tolist()

    prices_all = (
        st.session_state.precios_fase1
    )

    # Si por alguna razón los precios ya no están
    # disponibles, se vuelven a descargar.
    if prices_all is None:
        with st.spinner(
            "Recuperando precios..."
        ):
            prices_all = download_prices(
                tuple(tickers)
            )

    missing = [
        t
        for t in tickers
        if t not in prices_all.columns
    ]

    if missing:
        st.error(
            "Faltan precios para: "
            + ", ".join(missing)
        )
        st.stop()

    prices = prices_all[
        tickers
    ].copy()

    returns, mean_returns, cov_matrix, corr_matrix = (
        annual_statistics(prices)
    )

    if len(returns) < 100:
        st.error(
            "No hay suficientes observaciones comunes."
        )
        st.stop()

    # --------------------------------------------------------
    # 2.1 UNIVERSO
    # --------------------------------------------------------
    st.subheader(
        "2.1 Universo utilizado"
    )

    st.dataframe(
        universe,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # 2.2 ESTADÍSTICAS
    # --------------------------------------------------------
    st.divider()
    st.subheader(
        "2.2 Rendimiento esperado y volatilidad"
    )

    individual = pd.DataFrame({
        "Ticker": mean_returns.index,
        "Rendimiento esperado":
            mean_returns.values,
        "Volatilidad":
            np.sqrt(
                np.diag(
                    cov_matrix.values
                )
            ),
    })

    individual = individual.merge(
        universe,
        on="Ticker",
        how="left",
    )

    individual_show = (
        individual.copy()
    )

    individual_show[
        "Rendimiento esperado"
    ] *= 100

    individual_show[
        "Volatilidad"
    ] *= 100

    st.dataframe(
        individual_show[
            [
                "Ticker",
                "Nombre",
                "Sector",
                "Rendimiento esperado",
                "Volatilidad",
            ]
        ],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rendimiento esperado":
                st.column_config.NumberColumn(
                    format="%.2f %%"
                ),
            "Volatilidad":
                st.column_config.NumberColumn(
                    format="%.2f %%"
                ),
        },
    )

    # --------------------------------------------------------
    # 2.3 CORRELACIÓN Y COVARIANZA
    # --------------------------------------------------------
    st.divider()
    st.subheader(
        "2.3 Correlaciones y covarianzas"
    )

    fig_corr = px.imshow(
        corr_matrix,
        text_auto=".2f",
        aspect="auto",
        zmin=-1,
        zmax=1,
        title=(
            "Matriz de correlación "
            "de rendimientos diarios"
        ),
    )

    st.plotly_chart(
        fig_corr,
        use_container_width=True,
    )

    with st.expander(
        "Ver matriz de covarianzas anualizada"
    ):
        st.dataframe(
            cov_matrix,
            use_container_width=True,
        )

    # --------------------------------------------------------
    # 2.4 OPTIMIZACIÓN
    # --------------------------------------------------------
    st.divider()
    st.subheader(
        "2.4 Optimización de portafolios"
    )

    w_min = min_variance_portfolio(
        mean_returns,
        cov_matrix,
        risk_free,
    )

    w_sharpe = max_sharpe_portfolio(
        mean_returns,
        cov_matrix,
        risk_free,
    )

    w_equal = np.repeat(
        1 / len(tickers),
        len(tickers),
    )

    perf_min = portfolio_performance(
        w_min,
        mean_returns,
        cov_matrix,
        risk_free,
    )

    perf_sharpe = portfolio_performance(
        w_sharpe,
        mean_returns,
        cov_matrix,
        risk_free,
    )

    perf_equal = portfolio_performance(
        w_equal,
        mean_returns,
        cov_matrix,
        risk_free,
    )

    st.session_state.risk_free_markowitz = risk_free
    st.session_state.markowitz_resultados = {
        "tickers": tickers,
        "mean_returns": mean_returns,
        "cov_matrix": cov_matrix,
        "corr_matrix": corr_matrix,
        "w_sharpe": np.asarray(w_sharpe),
        "w_min": np.asarray(w_min),
        "perf_sharpe": perf_sharpe,
        "perf_min": perf_min,
        "perf_equal": perf_equal,
    }

    with st.spinner(
        f"Simulando {simulations:,} portafolios..."
    ):
        mc = monte_carlo(
            mean_returns,
            cov_matrix,
            risk_free,
            simulations,
        )

        frontier = efficient_frontier(
            mean_returns,
            cov_matrix,
            risk_free,
        )

    mc_plot = mc.copy()

    mc_plot["Rendimiento (%)"] = (
        mc_plot["Rendimiento"] * 100
    )

    mc_plot["Volatilidad (%)"] = (
        mc_plot["Volatilidad"] * 100
    )

    fig = px.scatter(
        mc_plot,
        x="Volatilidad (%)",
        y="Rendimiento (%)",
        color="Sharpe",
        opacity=0.40,
        title=(
            f"Markowitz — "
            f"{simulations:,} portafolios"
        ),
    )

    if not frontier.empty:
        fig.add_trace(
            go.Scatter(
                x=(
                    frontier["Volatilidad"]
                    * 100
                ),
                y=(
                    frontier["Rendimiento"]
                    * 100
                ),
                mode="lines",
                name="Frontera eficiente",
                line=dict(width=4),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[perf_min[1] * 100],
            y=[perf_min[0] * 100],
            mode="markers",
            marker=dict(
                size=16,
                symbol="diamond",
            ),
            name="Mínima varianza",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[perf_sharpe[1] * 100],
            y=[perf_sharpe[0] * 100],
            mode="markers",
            marker=dict(
                size=18,
                symbol="star",
            ),
            name="Máximo Sharpe",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[perf_equal[1] * 100],
            y=[perf_equal[0] * 100],
            mode="markers",
            marker=dict(
                size=14,
                symbol="circle-open",
            ),
            name="Equal Weight",
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # 2.5 RESULTADOS
    # --------------------------------------------------------
    st.divider()
    st.subheader(
        "2.5 Comparación de resultados"
    )

    comparison = pd.DataFrame([
        {
            "Portafolio":
                "Máximo Sharpe",
            "Rendimiento esperado":
                perf_sharpe[0],
            "Volatilidad":
                perf_sharpe[1],
            "Sharpe":
                perf_sharpe[2],
        },
        {
            "Portafolio":
                "Mínima varianza",
            "Rendimiento esperado":
                perf_min[0],
            "Volatilidad":
                perf_min[1],
            "Sharpe":
                perf_min[2],
        },
        {
            "Portafolio":
                "Equal Weight",
            "Rendimiento esperado":
                perf_equal[0],
            "Volatilidad":
                perf_equal[1],
            "Sharpe":
                perf_equal[2],
        },
    ])

    comparison_show = (
        comparison.copy()
    )

    comparison_show[
        "Rendimiento esperado"
    ] *= 100

    comparison_show[
        "Volatilidad"
    ] *= 100

    st.dataframe(
        comparison_show,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rendimiento esperado":
                st.column_config.NumberColumn(
                    format="%.2f %%"
                ),
            "Volatilidad":
                st.column_config.NumberColumn(
                    format="%.2f %%"
                ),
            "Sharpe":
                st.column_config.NumberColumn(
                    format="%.3f"
                ),
        },
    )

    # --------------------------------------------------------
    # 2.6 PESOS
    # --------------------------------------------------------
    st.divider()
    st.subheader(
        "2.6 Distribución del capital"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            "### Máximo Sharpe"
        )

        sharpe_weights = weights_table(
            w_sharpe,
            universe,
        )

        sharpe_weights[
            "Peso"
        ] *= 100

        st.dataframe(
            sharpe_weights,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Peso":
                    st.column_config.NumberColumn(
                        format="%.2f %%"
                    ),
            },
        )

    with col2:
        st.markdown(
            "### Mínima varianza"
        )

        min_weights = weights_table(
            w_min,
            universe,
        )

        min_weights[
            "Peso"
        ] *= 100

        st.dataframe(
            min_weights,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Peso":
                    st.column_config.NumberColumn(
                        format="%.2f %%"
                    ),
            },
        )

    # --------------------------------------------------------
    # 2.7 COMPARACIÓN POR SECTOR
    # --------------------------------------------------------
    st.divider()
    st.subheader(
        "2.7 Tecnología vs Industriales vs combinación"
    )

    tech = universe.loc[
        universe["Sector"]
        == "Tecnología",
        "Ticker",
    ].tolist()

    industrial = universe.loc[
        universe["Sector"]
        == "Industriales",
        "Ticker",
    ].tolist()

    tech_result = analyze_subset(
        tech,
        prices,
        risk_free,
    )

    ind_result = analyze_subset(
        industrial,
        prices,
        risk_free,
    )

    sectors = pd.DataFrame([
        {
            "Universo":
                "Solo Tecnología",
            "Rendimiento Máx. Sharpe":
                tech_result["sharpe"][0],
            "Volatilidad Máx. Sharpe":
                tech_result["sharpe"][1],
            "Sharpe máximo":
                tech_result["sharpe"][2],
            "Volatilidad mínima":
                tech_result["min"][1],
        },
        {
            "Universo":
                "Solo Industriales",
            "Rendimiento Máx. Sharpe":
                ind_result["sharpe"][0],
            "Volatilidad Máx. Sharpe":
                ind_result["sharpe"][1],
            "Sharpe máximo":
                ind_result["sharpe"][2],
            "Volatilidad mínima":
                ind_result["min"][1],
        },
        {
            "Universo":
                "Tecnología + Industriales",
            "Rendimiento Máx. Sharpe":
                perf_sharpe[0],
            "Volatilidad Máx. Sharpe":
                perf_sharpe[1],
            "Sharpe máximo":
                perf_sharpe[2],
            "Volatilidad mínima":
                perf_min[1],
        },
    ])

    sectors_show = (
        sectors.copy()
    )

    for col in [
        "Rendimiento Máx. Sharpe",
        "Volatilidad Máx. Sharpe",
        "Volatilidad mínima",
    ]:
        sectors_show[col] *= 100

    st.dataframe(
        sectors_show,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rendimiento Máx. Sharpe":
                st.column_config.NumberColumn(
                    format="%.2f %%"
                ),
            "Volatilidad Máx. Sharpe":
                st.column_config.NumberColumn(
                    format="%.2f %%"
                ),
            "Sharpe máximo":
                st.column_config.NumberColumn(
                    format="%.3f"
                ),
            "Volatilidad mínima":
                st.column_config.NumberColumn(
                    format="%.2f %%"
                ),
        },
    )

    # --------------------------------------------------------
    # 2.8 INTERPRETACIÓN
    # --------------------------------------------------------
    st.divider()
    st.subheader(
        "2.8 Interpretación"
    )

    st.markdown(
        """
        El modelo de Markowitz analiza simultáneamente el
        **rendimiento esperado**, la **volatilidad** y la
        **covarianza** entre los activos.

        - **Máximo Sharpe:** busca la mayor compensación de
          rendimiento sobre la tasa libre de riesgo por unidad
          de volatilidad.
        - **Mínima varianza:** busca la combinación de menor
          volatilidad posible.
        - **Frontera eficiente:** representa los portafolios
          que ofrecen el mayor rendimiento esperado para cada
          nivel de riesgo.
        - **Equal Weight:** funciona como referencia al asignar
          el mismo capital a todas las acciones.
        """
    )

    st.warning(
        "Las acciones fueron seleccionadas retrospectivamente "
        "por su buen desempeño histórico. Por ello existe "
        "sesgo retrospectivo y de supervivencia. Los resultados "
        "no representan una predicción de rendimientos futuros."
    )

    # --------------------------------------------------------
    # DESCARGAS
    # --------------------------------------------------------
    st.divider()
    st.subheader(
        "Descargar resultados"
    )

    st.download_button(
        "⬇️ Resultados Markowitz CSV",
        data=(
            comparison_show
            .to_csv(index=False)
            .encode("utf-8-sig")
        ),
        file_name=(
            "resultados_markowitz.csv"
        ),
        mime="text/csv",
        use_container_width=True,
    )


    st.divider()
    nav_back, nav_next = st.columns(2)

    with nav_back:
        if st.button(
            "← Volver a Fase 1",
            key="fase2_volver_final",
            use_container_width=True,
        ):
            st.session_state.fase = 1
            st.rerun()

    with nav_next:
        if st.button(
            "Siguiente → Interpretación detallada",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.fase = 3
            st.rerun()


# ============================================================
# FASE 3
# ============================================================
elif st.session_state.fase == 3:
    universe = st.session_state.universo_final
    results = st.session_state.markowitz_resultados

    if universe is None or results is None:
        st.warning("Primero es necesario completar la Fase 2.")
        if st.button("← Ir a Fase 2"):
            st.session_state.fase = 2
            st.rerun()
        st.stop()

    st.header("Fase 3 — Interpretación detallada de Markowitz")
    st.write(
        "Esta fase traduce los resultados matemáticos de Markowitz a una "
        "lectura más directa del portafolio."
    )

    tickers = results["tickers"]
    mean_returns = results["mean_returns"]
    cov_matrix = results["cov_matrix"]
    w_sharpe = results["w_sharpe"]
    w_min = results["w_min"]
    perf_sharpe = results["perf_sharpe"]
    perf_min = results["perf_min"]
    perf_equal = results["perf_equal"]
    risk_free = st.session_state.risk_free_markowitz

    st.subheader("3.1 ¿Cuál es el portafolio más eficiente?")
    st.success(
        "Dentro de los portafolios calculados, el de **Máximo Sharpe** es "
        "el que ofrece la mayor compensación de rendimiento esperado por "
        "unidad de volatilidad sobre la tasa libre de riesgo. No significa "
        "que garantice el mayor rendimiento futuro."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Rendimiento esperado", f"{perf_sharpe[0] * 100:.2f} %")
    c2.metric("Volatilidad", f"{perf_sharpe[1] * 100:.2f} %")
    c3.metric("Sharpe", f"{perf_sharpe[2]:.3f}")

    best_weights = weights_table(w_sharpe, universe)
    best_weights_show = best_weights.copy()
    best_weights_show["Peso"] *= 100

    st.dataframe(
        best_weights_show,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Peso": st.column_config.NumberColumn(format="%.2f %%"),
        },
    )

    st.subheader("3.2 Interpretación de los términos")
    st.markdown(
        f"""
        - **Rendimiento esperado:** media anualizada de los rendimientos históricos
          utilizados por el modelo. En el portafolio de Máximo Sharpe es
          **{perf_sharpe[0] * 100:.2f}%**.
        - **Volatilidad:** desviación estándar anualizada del portafolio; representa
          cuánto han variado históricamente sus rendimientos. Aquí es
          **{perf_sharpe[1] * 100:.2f}%**.
        - **Sharpe:** exceso de rendimiento esperado sobre la tasa libre de riesgo
          dividido entre la volatilidad. La tasa utilizada fue
          **{risk_free * 100:.2f}%**.
        - **Covarianza:** indica cómo tienden a moverse conjuntamente dos activos.
          Es una pieza central de la diversificación.
        - **Correlación:** versión normalizada de la covarianza, entre -1 y +1.
        - **Frontera eficiente:** conjunto de portafolios para los que no se puede
          aumentar el rendimiento esperado sin aceptar más volatilidad, o reducir
          volatilidad sin sacrificar rendimiento esperado.
        - **Mínima varianza:** portafolio de menor volatilidad posible con las
          restricciones establecidas.
        - **Equal Weight:** referencia en la que todos los activos reciben el mismo peso.
        """
    )

    st.subheader("3.3 Comparación de las tres referencias")
    phase3_comp = pd.DataFrame([
        {
            "Portafolio": "Máximo Sharpe",
            "Rendimiento (%)": perf_sharpe[0] * 100,
            "Volatilidad (%)": perf_sharpe[1] * 100,
            "Sharpe": perf_sharpe[2],
        },
        {
            "Portafolio": "Mínima varianza",
            "Rendimiento (%)": perf_min[0] * 100,
            "Volatilidad (%)": perf_min[1] * 100,
            "Sharpe": perf_min[2],
        },
        {
            "Portafolio": "Equal Weight",
            "Rendimiento (%)": perf_equal[0] * 100,
            "Volatilidad (%)": perf_equal[1] * 100,
            "Sharpe": perf_equal[2],
        },
    ])

    st.dataframe(
        phase3_comp,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rendimiento (%)": st.column_config.NumberColumn(format="%.2f %%"),
            "Volatilidad (%)": st.column_config.NumberColumn(format="%.2f %%"),
            "Sharpe": st.column_config.NumberColumn(format="%.3f"),
        },
    )

    st.info(
        "La Fase 4 no sustituirá Markowitz. Black-Litterman partirá de una "
        "expectativa de equilibrio del mercado y añadirá una opinión relativa "
        "con el nivel de confianza que indiques."
    )

    back3, next3 = st.columns(2)
    with back3:
        if st.button("← Regresar a Fase 2", use_container_width=True):
            st.session_state.fase = 2
            st.rerun()
    with next3:
        if st.button(
            "Siguiente → Black-Litterman",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.fase = 4
            st.rerun()


# ============================================================
# FASE 4
# ============================================================
elif st.session_state.fase == 4:
    universe = st.session_state.universo_final
    prices_all = st.session_state.precios_fase1
    markowitz = st.session_state.markowitz_resultados

    if universe is None or prices_all is None or markowitz is None:
        st.warning("Primero es necesario completar las fases anteriores.")
        if st.button("← Ir a Fase 2"):
            st.session_state.fase = 2
            st.rerun()
        st.stop()

    st.header("Fase 4 — Modelo Black-Litterman")
    st.write(
        "Black-Litterman combina los rendimientos de equilibrio implícitos en "
        "el mercado con **una opinión relativa** y su nivel de confianza."
    )

    tickers = universe["Ticker"].tolist()
    prices = prices_all[tickers].copy()
    _, historical_mu, cov_matrix, _ = annual_statistics(prices)

    with st.spinner("Obteniendo capitalizaciones bursátiles..."):
        market_caps = get_market_caps(tuple(tickers))

    missing_caps = market_caps[~np.isfinite(market_caps) | (market_caps <= 0)].index.tolist()
    if missing_caps:
        st.error(
            "No fue posible obtener una capitalización bursátil válida para: "
            + ", ".join(missing_caps)
            + ". Black-Litterman necesita esos pesos de mercado."
        )
        st.stop()

    market_weights = market_caps / market_caps.sum()
    market_weights = market_weights.reindex(tickers)

    auto_rf = get_auto_risk_free_rate()

    st.sidebar.header("Fase 4 — Black-Litterman")
    risk_free_pct_bl = st.sidebar.number_input(
        "Tasa libre de riesgo anual (%)",
        min_value=0.0,
        max_value=20.0,
        value=float(round(auto_rf * 100, 2)),
        step=0.05,
        help="Se precarga automáticamente con ^IRX como aproximación a Treasury de corto plazo, pero puedes editarla.",
    )
    tau = st.sidebar.number_input(
        "Tau (τ)",
        min_value=0.001,
        max_value=1.0,
        value=0.05,
        step=0.01,
        format="%.3f",
        help="Controla la incertidumbre sobre los rendimientos de equilibrio.",
    )

    risk_free_bl = risk_free_pct_bl / 100.0

    market_ret_hist = float(np.dot(market_weights.values, historical_mu.reindex(tickers).values))
    market_var = float(market_weights.values.T @ cov_matrix.loc[tickers, tickers].values @ market_weights.values)

    if market_var <= 0:
        st.error("No fue posible estimar una varianza de mercado válida.")
        st.stop()

    delta = (market_ret_hist - risk_free_bl) / market_var
    if delta <= 0:
        st.warning(
            "La aversión al riesgo estimada resultó no positiva con los datos y "
            "la tasa libre de riesgo seleccionada. Se usará un mínimo técnico de 0.01."
        )
        delta = 0.01

    pi_excess = pd.Series(
        delta * (cov_matrix.loc[tickers, tickers].values @ market_weights.values),
        index=tickers,
        name="Pi excess",
    )

    st.subheader("4.1 Parámetros de equilibrio")
    a, b, c, d = st.columns(4)
    a.metric("Tasa libre de riesgo", f"{risk_free_bl * 100:.2f} %")
    b.metric("τ", f"{tau:.3f}")
    c.metric("Aversión al riesgo δ", f"{delta:.3f}")
    d.metric("Activos", len(tickers))

    market_table = universe[["Ticker", "Nombre", "Sector"]].copy()
    market_table["Capitalización"] = market_caps.reindex(market_table["Ticker"]).values
    market_table["Peso mercado"] = market_weights.reindex(market_table["Ticker"]).values * 100
    market_table["π equilibrio (%)"] = pi_excess.reindex(market_table["Ticker"]).values * 100

    st.dataframe(
        market_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Capitalización": st.column_config.NumberColumn(format="$ %.0f"),
            "Peso mercado": st.column_config.NumberColumn(format="%.2f %%"),
            "π equilibrio (%)": st.column_config.NumberColumn(format="%.2f %%"),
        },
    )

    st.subheader("4.2 Introducir una opinión relativa")
    col_a, col_b = st.columns(2)
    with col_a:
        outperformer = st.selectbox(
            "Activo que creo que rendirá MÁS",
            options=tickers,
            index=0,
        )
    with col_b:
        under_options = [t for t in tickers if t != outperformer]
        underperformer = st.selectbox(
            "Activo que creo que rendirá MENOS",
            options=under_options,
            index=0,
        )

    col_spread, col_conf = st.columns(2)
    with col_spread:
        spread_pct = st.number_input(
            "¿Cuántos puntos porcentuales más al año?",
            min_value=0.01,
            max_value=100.0,
            value=5.0,
            step=0.25,
        )
    with col_conf:
        confidence_pct = st.number_input(
            "Confianza en la opinión (%)",
            min_value=1.0,
            max_value=99.9,
            value=70.0,
            step=1.0,
        )

    st.info(
        f"Opinión: **{outperformer} superará a {underperformer} en "
        f"{spread_pct:.2f} puntos porcentuales anuales**, con "
        f"**{confidence_pct:.1f}% de confianza**."
    )

    p, q, omega = black_litterman_relative_view(
        cov_matrix=cov_matrix,
        market_weights=market_weights,
        risk_free=risk_free_bl,
        tau=tau,
        outperformer=outperformer,
        underperformer=underperformer,
        spread=spread_pct / 100.0,
        confidence=confidence_pct / 100.0,
    )

    posterior_excess = black_litterman_posterior(
        cov_matrix=cov_matrix,
        pi_excess=pi_excess,
        tau=tau,
        p=p,
        q=q,
        omega=omega,
    )

    posterior_total = posterior_excess + risk_free_bl
    equilibrium_total = pi_excess + risk_free_bl

    w_bl = max_sharpe_portfolio(
        posterior_total,
        cov_matrix,
        risk_free_bl,
    )
    perf_bl = portfolio_performance(
        w_bl,
        posterior_total,
        cov_matrix,
        risk_free_bl,
    )

    # Recalcular Markowitz con la misma tasa libre de riesgo de Fase 4 para
    # que la comparación sea homogénea.
    w_markowitz_compare = max_sharpe_portfolio(
        historical_mu,
        cov_matrix,
        risk_free_bl,
    )
    perf_markowitz_compare = portfolio_performance(
        w_markowitz_compare,
        historical_mu,
        cov_matrix,
        risk_free_bl,
    )

    st.subheader("4.3 Rendimientos: equilibrio vs Black-Litterman")
    returns_compare = universe[["Ticker", "Nombre", "Sector"]].copy()
    returns_compare["Equilibrio π (%)"] = equilibrium_total.reindex(tickers).values * 100
    returns_compare["BL posterior (%)"] = posterior_total.reindex(tickers).values * 100
    returns_compare["Cambio (pp)"] = (
        posterior_total.reindex(tickers).values
        - equilibrium_total.reindex(tickers).values
    ) * 100

    st.dataframe(
        returns_compare,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Equilibrio π (%)": st.column_config.NumberColumn(format="%.2f %%"),
            "BL posterior (%)": st.column_config.NumberColumn(format="%.2f %%"),
            "Cambio (pp)": st.column_config.NumberColumn(format="%+.2f"),
        },
    )

    st.subheader("4.4 Markowitz vs Black-Litterman")
    weights_compare = universe[["Ticker", "Nombre", "Sector"]].copy()
    weights_compare["Markowitz"] = w_markowitz_compare * 100
    weights_compare["Black-Litterman"] = w_bl * 100
    weights_compare["Cambio (pp)"] = (w_bl - w_markowitz_compare) * 100

    st.dataframe(
        weights_compare.sort_values("Black-Litterman", ascending=False),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Markowitz": st.column_config.NumberColumn(format="%.2f %%"),
            "Black-Litterman": st.column_config.NumberColumn(format="%.2f %%"),
            "Cambio (pp)": st.column_config.NumberColumn(format="%+.2f"),
        },
    )

    graph_weights = weights_compare.melt(
        id_vars=["Ticker"],
        value_vars=["Markowitz", "Black-Litterman"],
        var_name="Modelo",
        value_name="Peso (%)",
    )

    fig_bl = px.bar(
        graph_weights,
        x="Ticker",
        y="Peso (%)",
        color="Modelo",
        barmode="group",
        title="Distribución del capital — Markowitz vs Black-Litterman",
    )
    st.plotly_chart(fig_bl, use_container_width=True)

    perf_compare = pd.DataFrame([
        {
            "Modelo": "Markowitz — Máximo Sharpe",
            "Rendimiento esperado (%)": perf_markowitz_compare[0] * 100,
            "Volatilidad (%)": perf_markowitz_compare[1] * 100,
            "Sharpe": perf_markowitz_compare[2],
        },
        {
            "Modelo": "Black-Litterman — Máximo Sharpe",
            "Rendimiento esperado (%)": perf_bl[0] * 100,
            "Volatilidad (%)": perf_bl[1] * 100,
            "Sharpe": perf_bl[2],
        },
    ])

    st.dataframe(
        perf_compare,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rendimiento esperado (%)": st.column_config.NumberColumn(format="%.2f %%"),
            "Volatilidad (%)": st.column_config.NumberColumn(format="%.2f %%"),
            "Sharpe": st.column_config.NumberColumn(format="%.3f"),
        },
    )

    st.subheader("4.5 Interpretación automática")
    change_series = pd.Series(w_bl - w_markowitz_compare, index=tickers)
    largest_up = change_series.idxmax()
    largest_down = change_series.idxmin()

    st.markdown(
        f"""
        Tu opinión favorece a **{outperformer}** frente a **{underperformer}** por
        **{spread_pct:.2f} pp anuales**, con una confianza de
        **{confidence_pct:.1f}%**.

        Black-Litterman no sustituye por completo las expectativas implícitas del
        mercado: las combina con esa opinión según su incertidumbre. Con los
        parámetros actuales, el mayor aumento de peso frente a Markowitz corresponde
        a **{largest_up}** ({change_series[largest_up] * 100:+.2f} pp) y la mayor
        reducción corresponde a **{largest_down}**
        ({change_series[largest_down] * 100:+.2f} pp).

        Un nivel de confianza mayor reduce la incertidumbre de la opinión (Ω) y hace
        que el resultado posterior responda con mayor intensidad a tu view. Un nivel
        de confianza menor conserva más peso de las expectativas de equilibrio.
        """
    )

    with st.expander("Ver matrices internas P, Q y Ω"):
        st.write("**P — vector de la opinión relativa**")
        st.dataframe(pd.DataFrame(p, columns=tickers), use_container_width=True)
        st.write("**Q — magnitud de la opinión**")
        st.dataframe(pd.DataFrame({"Q": q}), use_container_width=True)
        st.write("**Ω — incertidumbre de la opinión**")
        st.dataframe(pd.DataFrame(omega, columns=["Ω"], index=["View 1"]), use_container_width=True)

    st.warning(
        "Black-Litterman sigue dependiendo de estimaciones, supuestos y datos "
        "históricos. La opinión introducida es un escenario del usuario, no una "
        "predicción generada por el modelo."
    )

    st.download_button(
        "⬇️ Comparación Black-Litterman CSV",
        data=weights_compare.to_csv(index=False).encode("utf-8-sig"),
        file_name="comparacion_markowitz_black_litterman.csv",
        mime="text/csv",
        use_container_width=True,
    )

    back4, restart4 = st.columns(2)
    with back4:
        if st.button("← Regresar a Fase 3", use_container_width=True):
            st.session_state.fase = 3
            st.rerun()
    with restart4:
        if st.button("Volver a Fase 1", use_container_width=True):
            st.session_state.fase = 1
            st.rerun()
