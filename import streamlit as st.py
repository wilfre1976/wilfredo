import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, max_error

st.set_page_config(page_title="Barra 1D — Analítica + ML", layout="wide")

DEFAULTS = {"L": 0.80, "k": 1.17e-4, "T1": 10.0, "T2": 60.0, "Ti": 23.0}

@st.cache_data
def compute_Bn(N_terms, L, Ti, T1, T2):
    xi = np.linspace(0.0, L, 2000)
    u0 = lambda xv: T1 + (T2 - T1) / L * xv
    Bn = np.zeros(N_terms)
    for n in range(1, N_terms + 1):
        lam = n * np.pi / L
        f = (Ti - u0(xi)) * np.sin(lam * xi)
        Bn[n-1] = (2.0 / L) * simpson(f, x=xi)
    return Bn

@st.cache_data
def build_solution(L, k, T1, T2, Ti, N_terms, Nx, Nt, t_max):
    x = np.linspace(0.0, L, Nx)
    t = np.linspace(0.0, t_max, Nt)
    Bn = compute_Bn(N_terms, L, Ti, T1, T2)
    lambda_n = np.arange(1, N_terms + 1) * np.pi / L
    sin_nx = np.sin(np.outer(lambda_n, x))                       # (N_terms, Nx)
    exp_nt = np.exp(-k * (lambda_n**2)[:, None] * t[None, :])    # (N_terms, Nt)
    u0 = T1 + (T2 - T1) / L * x                                   # (Nx,)
    U = u0[:, None] + sin_nx.T.dot(Bn[:, None] * exp_nt)          # (Nx, Nt)
    return x, t, U, Bn

st.title("Barra 1D — Solución analítica + ML")

with st.sidebar:
    st.header("Parámetros")
    L = st.number_input("L (m)", value=DEFAULTS["L"], format="%.3f")
    k = st.number_input("k (m²/s)", value=DEFAULTS["k"], format="%.6e")
    T1 = st.number_input("T1 (°C)", value=DEFAULTS["T1"])
    T2 = st.number_input("T2 (°C)", value=DEFAULTS["T2"])
    Ti = st.number_input("Ti (°C)", value=DEFAULTS["Ti"])
    N_terms = st.slider("Términos Fourier N", 10, 300, 100, 10)
    Nx = st.slider("Puntos espaciales Nx", 50, 500, 200, 10)
    Nt = st.slider("Puntos temporales Nt", 50, 800, 300, 25)
    t_max = st.slider("Tiempo total (s)", 10, 3600, 300, 10)
    run_ml = st.checkbox("Entrenar ML (puede tardar)", value=False)
    ml_subsample = st.slider("Máx muestras ML", 2000, 50000, 20000, 1000)
    if st.button("Recalcular"):
        compute_Bn.clear(); build_solution.clear(); st.experimental_rerun()

with st.spinner("Calculando..."):
    x, t, U, Bn = build_solution(L, k, T1, T2, Ti, N_terms, Nx, Nt, t_max)

x_cm = x * 100
st.subheader("Mapa u(x,t)")
fig = plt.figure(figsize=(9,4))
ax = fig.add_subplot(111)
im = ax.imshow(U.T, aspect='auto', origin='lower', extent=[0, L*100, 0, t_max], cmap='plasma')
ax.set_xlabel("x (cm)"); ax.set_ylabel("t (s)")
fig.colorbar(im, ax=ax, label='°C')
st.pyplot(fig)

st.subheader("Perfiles espaciales")
default_times = [0, 5, 15, 30, 60, 120, int(t_max)]
times = st.multiselect("Tiempos (s)", default_times, default=[0, 30, int(t_max)])
fig2, ax2 = plt.subplots(figsize=(8,4))
for tt in times:
    idx = np.argmin(np.abs(t - tt))
    ax2.plot(x_cm, U[:, idx], label=f"t={tt}s")
ax2.plot(x_cm, T1 + (T2 - T1)/L * x, '--k', label='Estacionario')
ax2.set_xlabel("x (cm)"); ax2.set_ylabel("°C"); ax2.legend()
st.pyplot(fig2)

if run_ml:
    st.subheader("ML — entrenamiento (submuestreo)")
    X_ml = np.array([(xi, tj) for xi in x for tj in t])
    y_ml = U.flatten()
    n_total = len(y_ml)
    if n_total > ml_subsample:
        idxs = np.random.choice(n_total, ml_subsample, replace=False)
        Xs, ys = X_ml[idxs], y_ml[idxs]
    else:
        Xs, ys = X_ml, y_ml
    X_train, X_test, y_train, y_test = train_test_split(Xs, ys, test_size=0.2, random_state=42)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train); X_test_s = scaler.transform(X_test)
    lr = LinearRegression().fit(X_train_s, y_train); y_lr = lr.predict(X_test_s)
    poly = PolynomialFeatures(degree=3); Xtr_poly = poly.fit_transform(X_train_s); Xte_poly = poly.transform(X_test_s)
    lr_poly = LinearRegression().fit(Xtr_poly, y_train); y_poly = lr_poly.predict(Xte_poly)
    mlp = MLPRegressor(hidden_layer_sizes=(50,25), max_iter=300, early_stopping=True, random_state=42)
    mlp.fit(X_train_s, y_train); y_mlp = mlp.predict(X_test_s)
    def safe_mape(y_true, y_pred):
        den = np.where(y_true == 0, np.nan, y_true)
        return np.nanmean(np.abs((y_true - y_pred) / den)) * 100
    rows = []
    for name, y_pred in [("Lineal", y_lr), ("Polinomial3", y_poly), ("MLP", y_mlp)]:
        r2 = r2_score(y_test, y_pred); mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred)); mx = max_error(y_test, y_pred)
        mape = safe_mape(y_test, y_pred)
        rows.append({"Modelo": name, "R2": f"{r2:.4f}", "MAE": f"{mae:.4f}", "RMSE": f"{rmse:.4f}", "MAPE(%)": f"{mape:.4f}", "MaxErr": f"{mx:.4f}"})
    st.table(rows)

st.info("Ejecutar: C:\\Program Files\\Python314\\python.exe -m streamlit run \"c:\\Users\\BIB.LUS\\Desktop\\app_streamlit.py\"")