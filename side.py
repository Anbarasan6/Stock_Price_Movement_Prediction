import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 🏗 Set Page Layout
st.set_page_config(page_title="Stock Price Prediction", layout="wide")

# 🎨 **Custom Styling (Background & Sidebar)**
def set_bg():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: skyblue !important;
            color: black !important;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(to bottom, black 5.3%, #2c3e50 5%); /* 5% Black, Rest Dark Blue */
            color: white;
            padding: 15px; /* Extra spacing */
        }

        /* Increase Font Size for Sidebar Elements */
        [data-testid="stSidebar"] * {
            font-size: 20px !important;
        }

        /* Make Sidebar Titles Bigger */
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3 {
            font-size: 28px !important;
            font-weight: bold;
            text-align: center;
        }

        /* Adjust Input Fields & Buttons */
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
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg()

# 📊 Streamlit UI
st.markdown("<h1 style='text-align: center; color: white;'>📈 Stock Price Movement Prediction</h1>", unsafe_allow_html=True)

# 🏠 **Sidebar for File Upload and Manual Prediction**
with st.sidebar:
    st.subheader("📂 Upload Dataset")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    st.subheader("🔮 Predict Stock Movement Manually")
    open_price = st.number_input("💰 Open Price", value=0.0, format="%.2f")
    close_price = st.number_input("📉 Close Price", value=0.0, format="%.2f")
    high_price = st.number_input("📈 High Price", value=0.0, format="%.2f")
    low_price = st.number_input("📉 Low Price", value=0.0, format="%.2f")
    is_quarter_end = st.selectbox("📆 Is it Quarter End?", [0, 1])

    predict_button = st.button("🔍 Predict")

# ✅ **Load Default Dataset (if no file uploaded)**
if uploaded_file is None:
    df = pd.read_csv("nifty50_data.csv")
else:
    df = pd.read_csv(uploaded_file)

df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

df = df.round(2)  # ✅ Round all numerical values to 2 decimal places


# 📆 **Filter Last One Year Data**
last_year = df['Date'].max() - pd.DateOffset(years=1)
df_last_year = df[df['Date'] >= last_year]

# 🔢 **Feature Engineering**
df['is_quarter_end'] = np.where(df['Date'].dt.month % 3 == 0, 1, 0)
df['open-close'] = df['Open'] - df['Close']
df['low-high'] = df['Low'] - df['High']
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)  # 1 = UP, 0 = DOWN

# 📊 **Stock Moving Line Chart (Last One Year)**
st.subheader("📈 Stock Price Trend (Last 1 Year)")

# Plot Close Price for the Last Year
fig = px.line(df_last_year, x="Date", y=["Close"], labels={"value": "Stock Price", "Date": "Date"})

# 📌 **Updated Chart Layout**
fig.update_layout(
    autosize=True,  
    plot_bgcolor="#d1ecf1",  # Light Blue Background
    paper_bgcolor="#3498db",  # Sky Blue Paper Background
    showlegend=False,  

    xaxis=dict(
        title="Date",
        title_font=dict(size=20, color="white"),
        tickfont=dict(size=16, color="black"),
        showgrid=True,
        gridcolor="white",
    ),
    
    yaxis=dict(
        title="Price",
        title_font=dict(size=20, color="white"),
        tickfont=dict(size=16, color="black"),
        showgrid=True,
        gridcolor="white",
    ),

    margin=dict(l=0, r=0, t=0, b=0),  # Remove margins
)

# **White Line with Thickness**
fig.update_traces(line=dict(color="blue", width=2))

# 📊 **Display Chart**
st.plotly_chart(fig, use_container_width=True)

# 📋 **Display Last 5 Records of Feature Values**
st.subheader("📋 Last 5 Records of Stock Data")

# 🎨 **Stylish Table**
st.markdown(
    """
    <style>
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

# Convert the last 5 records to an HTML table
last_5_records = df[['Open', 'Close', 'High', 'Low', 'is_quarter_end', 'target']].tail(5)
styled_table = last_5_records.to_html(index=False, escape=False)

# Display the styled table
st.markdown(styled_table, unsafe_allow_html=True)

# 🏗 **Select Features**
features = ['open-close', 'low-high', 'is_quarter_end']
target = 'target'

# ✂️ **Split Data**
#X_train, X_test, Y_train, Y_test = train_test_split(df[features], df[target],random_state=42)
X_train = df[features]
Y_train = df[target]


# 🔧 **Scale Features**
scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)

# 🚀 **Train ML Model (SVM)**
model = SVC(kernel='rbf', C=1.0, random_state=42)
model.fit(X_train, Y_train)

# 🔮 **Make Prediction**
if predict_button:
    manual_data = np.array([[open_price - close_price, low_price - high_price, is_quarter_end]])
    
    # Convert to DataFrame with same feature names
    manual_df = pd.DataFrame(manual_data, columns=['open-close', 'low-high', 'is_quarter_end'])
    
    # Apply scaling
    #manual_data_scaled = scaler.transform(manual_df)

    # Predict
    prediction = model.predict(manual_data)

    if prediction[0] == 1:
        st.success("✅ Stock will go UP 📈")
        st.markdown("<h3 style='color: green; text-align: center;'>📊 Bullish Trend Expected! 🚀</h3>", unsafe_allow_html=True)
    else:
        st.error("❌ Stock will go DOWN 📉")
        st.markdown("<h3 style='color: red; text-align: center;'>📉 Bearish Trend Expected! ⚠️</h3>", unsafe_allow_html=True)
