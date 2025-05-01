import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 🏗 Set Page Layout
st.set_page_config(page_title="Stock Price Classifier", layout="wide")

# 🎨 Custom Styling (Background, Sidebar, Table)
def set_bg():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: skyblue !important;
            color: black !important;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(to bottom, black 5.3%, #2c3e50 5%);
            color: white;
            padding: 15px;
        }

        [data-testid="stSidebar"] * {
            font-size: 20px !important;
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            font-size: 28px !important;
            font-weight: bold;
            text-align: center;
        }

        [data-testid="stSidebar"] .stButton>button {
            font-size: 18px !important;
            padding: 12px;
            width: 100%;
            font-weight: bold;
        }

        [data-testid="stSidebar"] .stTextInput>div>div>input,
        [data-testid="stSidebar"] .stNumberInput>div>div>input,
        [data-testid="stSidebar"] .stSelectbox>div>div>select {
            font-size: 18px !important;
            padding: 10px;
            width: 100%;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            border: 2px solid black;
        }
        th {
            background-color: #3498db;
            color: white;
            padding: 12px;
            font-size: 16px;
            text-align: center;
            border: 1px solid black;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        tr:nth-child(odd) {
            background-color: #d1ecf1;
        }
        td {
            padding: 10px;
            text-align: center;
            font-size: 14px;
            font-weight: bold;
            border: 1px solid black;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg()

st.markdown("<h1 style='text-align: center; color: white;'>📈 Stock Price Movement Classifier</h1>", unsafe_allow_html=True)

# 📂 Sidebar
with st.sidebar:
    st.subheader("📂 Upload Dataset")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    st.subheader("📅 Filter Data by Date")
    start_date = st.date_input("Start Date", pd.to_datetime("2023-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("2024-12-31"))

    st.subheader("🔮 Predict Stock Movement Manually")
    open_price = st.number_input("💰 Open Price", value=0.0, format="%.2f")
    close_price = st.number_input("📉 Close Price", value=0.0, format="%.2f")
    high_price = st.number_input("📈 High Price", value=0.0, format="%.2f")
    low_price = st.number_input("📉 Low Price", value=0.0, format="%.2f")
    is_quarter_end = st.selectbox("📆 Is it Quarter End?", [0, 1])

    predict_button = st.button("🔍 Predict")

# 📦 Load data
if uploaded_file is None:
    df = pd.read_csv("MSFT_realtime.csv")
else:
    df = pd.read_csv(uploaded_file)

df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df = df.round(2)

# 🧮 Feature Engineering
df['is_quarter_end'] = np.where(df['Date'].dt.month % 3 == 0, 1, 0)
df['open-close'] = df['Open'] - df['Close']
df['low-high'] = df['Low'] - df['High']
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

# 📆 Filter Date Range
filtered_df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]

# 📊 Chart
st.subheader("📈 Stock Price Trend")
fig = px.line(filtered_df, x="Date", y=["Close"], labels={"value": "Stock Price", "Date": "Date"})
fig.update_layout(
    autosize=True,
    plot_bgcolor="#d1ecf1",
    paper_bgcolor="#3498db",
    showlegend=False,
    xaxis=dict(title="Date", title_font=dict(size=20, color="white"), tickfont=dict(size=16, color="black"), showgrid=True, gridcolor="white"),
    yaxis=dict(title="Price", title_font=dict(size=20, color="white"), tickfont=dict(size=16, color="black"), showgrid=True, gridcolor="white"),
    margin=dict(l=0, r=0, t=0, b=0),
)
fig.update_traces(line=dict(color="blue", width=2))
st.plotly_chart(fig, use_container_width=True)

# 📋 Last 5 Records
st.subheader("📋 sample Records of Stock Data")
last_5_records = df[['Open', 'Close', 'High', 'Low', 'is_quarter_end', 'target']].tail(5)
styled_table = last_5_records.to_html(index=False, escape=False)
st.markdown(styled_table, unsafe_allow_html=True)

# 🏗 Model Setup
features = ['open-close', 'low-high', 'is_quarter_end']
target = 'target'

X = df[features]
y = df[target]

# ✅ Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🔧 Train Model
model = SVC(kernel='rbf', C=1.0, random_state=42)
model.fit(X_scaled, y)

# 🕘 Prediction History Session
if "pred_history" not in st.session_state:
    st.session_state.pred_history = []

# 🔮 Predict on Manual Input
if predict_button:
    input_data = np.array([[open_price - close_price, low_price - high_price, is_quarter_end]])
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]

    result = {
        "Open": open_price,
        "Close": close_price,
        "High": high_price,
        "Low": low_price,
        "is_quarter_end": is_quarter_end,
        "Prediction": "UP 📈" if prediction == 1 else "DOWN 📉"
    }
    st.session_state.pred_history.append(result)

    if prediction == 1:
        st.success("✅ Stock will go UP 📈")
        st.markdown("<h3 style='color: green; text-align: center;'>📊 Bullish Trend Expected! 🚀</h3>", unsafe_allow_html=True)
    else:
        st.error("❌ Stock will go DOWN 📉")
        st.markdown("<h3 style='color: red; text-align: center;'>📉 Bearish Trend Expected! ⚠️</h3>", unsafe_allow_html=True)

# 📜 Show Prediction History
if st.session_state.pred_history:
    st.subheader("🕘 Manual Prediction History")
    history_df = pd.DataFrame(st.session_state.pred_history)
    styled_history = history_df.to_html(index=False, escape=False)
    st.markdown(styled_history, unsafe_allow_html=True)
