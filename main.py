import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.interpolate import griddata

# Load dữ liệu
@st.cache_data
def load_data():
    df = pd.read_excel('du_lieu_frezing.xlsx')
    return df

df = load_data()

st.title("📊 Ứng dụng dự đoán độ nhám bề mặt Ra khi phay bằng MLP")

# Tiền xử lý
X = df[['V', 'S', 't']]
y = df['Ra']
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình
@st.cache_resource
def train_model():
    param_grid = {
        'hidden_layer_sizes': [(13,)],
        'learning_rate_init': [0.001],
        'activation': ['relu']
    }
    grid = GridSearchCV(
        estimator=MLPRegressor(max_iter=1000, early_stopping=True, random_state=42),
        param_grid=param_grid,
        cv=3,
        scoring='r2'
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_

model = train_model()
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Tabs giao diện
tab1, tab2, tab3 = st.tabs(["📈 Phân tích dữ liệu", "🎯 Tìm thông số phù hợp", "🔮 Dự đoán Ra"])

with tab1:
    st.subheader("1️⃣ Kết quả mô hình & biểu đồ")
    st.write(f"**Cấu hình mạng:** 13 neuron, activation='relu', learning_rate=0.001")
    st.write(f"**R² = {r2:.4f}, MSE = {mse:.4f}**")

    st.subheader("Bảng so sánh dự đoán")
    df_compare = pd.DataFrame({
        'Ra thực tế': y_test.reset_index(drop=True),
        'Ra dự đoán': y_pred
    })
    st.dataframe(df_compare)

    st.subheader("Biểu đồ: Ra thực tế vs dự đoán")
    fig1, ax1 = plt.subplots()
    ax1.scatter(y_test, y_pred, c='blue', label='Dự đoán vs Thực tế')
    ax1.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label='Đường lý tưởng')
    ax1.set_xlabel('Thực tế Ra (μm)')
    ax1.set_ylabel('Dự đoán Ra (μm)')
    ax1.legend()
    ax1.grid(True)
    st.pyplot(fig1)

    st.subheader("Biểu đồ: Ra theo chỉ số mẫu")
    fig2, ax2 = plt.subplots()
    ax2.plot(y_test.values, label='Ra thực tế', marker='o')
    ax2.plot(y_pred, label='Ra dự đoán', marker='x')
    ax2.set_xlabel('Chỉ số mẫu')
    ax2.set_ylabel('Ra (μm)')
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

    st.subheader("Biểu đồ miền: Ra theo (S, V)")
    x = df['S']
    yv = df['V']
    z = df['Ra']
    xi, yi = np.meshgrid(np.linspace(x.min(), x.max(), 100), np.linspace(yv.min(), yv.max(), 100))
    zi = griddata((x, yv), z, (xi, yi), method='cubic')
    fig3, ax3 = plt.subplots()
    cs = ax3.contourf(xi, yi, zi, levels=20, cmap='viridis')
    fig3.colorbar(cs, ax=ax3)
    ax3.set_xlabel('S')
    ax3.set_ylabel('V')
    st.pyplot(fig3)

with tab2:
    st.header("Tìm S, V, t cho Ra mong muốn")
    target_ra = st.number_input("Nhập giá trị Ra mong muốn (μm)", value=1.2, format="%.2f")
    n = st.slider("Số kết quả gợi ý", 1, 10, 4)

    if st.button("Tìm thông số"):
        V_range = np.linspace(df['V'].min(), df['V'].max(), 30)
        S_range = np.linspace(df['S'].min(), df['S'].max(), 30)
        t_range = np.linspace(df['t'].min(), df['t'].max(), 30)

        results = []
        for v in V_range:
            for s in S_range:
                for t in t_range:
                    row = pd.DataFrame([[v, s, t]], columns=['V', 'S', 't'])
                    row_scaled = scaler.transform(row)
                    ra = model.predict(row_scaled)[0]
                    sai_so = abs(ra - target_ra)
                    results.append((sai_so, v, s, t, ra))

        results.sort()
        top_k = results[:n]
        df_result = pd.DataFrame(top_k, columns=['Sai số', 'V', 'S', 't', 'Ra dự đoán'])
        st.dataframe(df_result.drop(columns='Sai số').reset_index(drop=True))

with tab3:
    st.header("Dự đoán Ra từ thông số cắt")
    s = st.number_input("S - Lượng chạy dao (mm/tooth)", value=0.15, format="%.3f")
    v = st.number_input("V - Tốc độ cắt (m/min)", value=120.0, format="%.1f")
    t = st.number_input("t - Chiều sâu cắt (mm)", value=0.5, format="%.3f")

    if st.button("Dự đoán"):
        input_df = pd.DataFrame([[v, s, t]], columns=['V', 'S', 't'])
        input_scaled = scaler.transform(input_df)
        ra_pred = model.predict(input_scaled)[0]
        st.success(f"✅ Dự đoán Ra: {ra_pred:.4f} μm")
