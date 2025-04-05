import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from scipy.interpolate import griddata
import io

st.set_page_config(page_title="Прогноз шероховатости Ra", layout="wide")
st.title("Приложение: Прогноз и анализ шероховатости поверхности Ra при торцевом фрезеровании заготовок из стали 20")
st.markdown("""
**Создатель**: Нгуен Нгок Шон - МТ3/МГТУ им. Баумана  
**Цель**: Прогноз значения Ra по технологическим параметрам и обратный поиск параметров по желаемому Ra при торцевом фрезеровании заготовок из стали 20.  
**Инструмент**: Торцевая фреза BAP300R-40-22 (D=40 мм, зубьев), пластины APMT1135PDER-M2 OP1215.
""")

@st.cache_data
def load_and_train_model():
    df = pd.read_excel("du_lieu_frezing.xlsx")
    X = df[['V', 'S', 't']]
    y = df['Ra']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    param_grid = {
        'hidden_layer_sizes': [(10,), (30,), (50,), (20, 10)],
        'learning_rate_init': [0.001, 0.005],
        'activation': ['relu']
    }
    grid = GridSearchCV(MLPRegressor(max_iter=5000, early_stopping=True, random_state=42),
                        param_grid, cv=3, scoring='r2', n_jobs=-1)
    grid.fit(X_train, y_train)
    model = grid.best_estimator_
    return df, scaler, model

df, scaler, model = load_and_train_model()

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Данные и графики", "🔍 Обратный поиск по Ra", "📈 Прогноз Ra"])

with tab1:
    st.subheader("📊 Таблицы и графики")
    st.dataframe(df)

    st.markdown("### График зависимости Ra от (S, V)")
    x, yv, z = df['S'], df['V'], df['Ra']
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(yv.min(), yv.max(), 100)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x, yv), z, (xi, yi), method='cubic')
    fig1, ax1 = plt.subplots()
    cp = ax1.contourf(xi, yi, zi, cmap='viridis')
    fig1.colorbar(cp)
    ax1.set_xlabel('S (мм/зуб)')
    ax1.set_ylabel('V (м/мин)')
    ax1.set_title('Ra по (S, V)')
    st.pyplot(fig1)

    st.markdown("### Обучающая кривая (пример)")
    if hasattr(model, 'loss_curve_'):
        fig2, ax2 = plt.subplots()
        ax2.plot(model.loss_curve_)
        ax2.set_title("Кривая обучения")
        ax2.set_xlabel("Эпохи")
        ax2.set_ylabel("Ошибка")
        st.pyplot(fig2)

with tab2:
    st.subheader("🔍 Обратный поиск по Ra")
    target_ra = st.number_input("Желаемое значение Ra (μm):", 0.1, 10.0, 1.2, 0.1)
    num_results = st.slider("Количество комбинаций для вывода:", 1, 10, 4)

    if st.button("🔎 Найти параметры"):
        V_range = np.linspace(df['V'].min(), df['V'].max(), 30)
        S_range = np.linspace(df['S'].min(), df['S'].max(), 30)
        t_range = np.linspace(df['t'].min(), df['t'].max(), 30)
        results = []
        for v in V_range:
            for s in S_range:
                for t in t_range:
                    input_df = pd.DataFrame([[v, s, t]], columns=['V', 'S', 't'])
                    input_scaled = scaler.transform(input_df)
                    ra = model.predict(input_scaled)[0]
                    err = abs(ra - target_ra)
                    results.append((err, v, s, t, ra))
        results.sort()
        out_df = pd.DataFrame(results[:num_results], columns=['Ошибка', 'V (м/мин)', 'S (мм/зуб)', 't (мм)', 'Ra прогноз'])
        st.dataframe(out_df)

with tab3:
    st.subheader("📈 Прогноз Ra")
    v = st.number_input("Скорость резания V (м/мин):", 10.0, 300.0, 120.0, 5.0)
    s = st.number_input("Подача на зуб S (мм/зуб):", 0.01, 1.0, 0.15, 0.01)
    t = st.number_input("Глубина резания t (мм):", 0.1, 5.0, 0.5, 0.1)

    if st.button("📉 Прогнозировать Ra"):
        input_df = pd.DataFrame([[v, s, t]], columns=['V', 'S', 't'])
        input_scaled = scaler.transform(input_df)
        ra_pred = model.predict(input_scaled)[0]
        st.success(f"Прогнозируемое Ra: {ra_pred:.4f} μm при V={v} м/мин, S={s} мм/зуб, t={t} мм")
