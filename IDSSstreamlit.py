import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression

# Configure page settings
st.set_page_config(
    page_title="Car Replacement Decision Support System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Helper Functions ===
def calculate_weighted_score(cost_score, style_score, reliability_score, fuel_economy_score, safety_score, weights):
    return (weights['cost'] * cost_score +
            weights['style'] * style_score +
            weights['reliability'] * reliability_score +
            weights['fuel_economy'] * fuel_economy_score +
            weights['safety'] * safety_score)

def calculate_pv(cash_flows, discount_rate=0.05):
    return sum([cf / (1 + discount_rate)**year for year, cf in enumerate(cash_flows, start=0)])

def predict_operating_costs(operating_costs):
    """
    Predict the operating costs for the next 5 years using linear regression.
    Input: `operating_costs` must have exactly 3 values.
    """
    if len(operating_costs) != 3:
        raise ValueError("Operating costs must have exactly 3 years of data.")
    past_years = np.array([1, 2, 3]).reshape(-1, 1)
    model = LinearRegression()
    model.fit(past_years, operating_costs)
    future_years = np.array([4, 5, 6, 7, 8]).reshape(-1, 1)
    return [max(0, cf) for cf in model.predict(future_years)]

def format_cash_flows(cash_flows):
    inflows = [cf if cf > 0 else 0 for cf in cash_flows]
    outflows = [abs(cf) if cf < 0 else 0 for cf in cash_flows]
    return inflows, outflows

def load_excel_data(file_path, sheet_name, required_columns):
    if not os.path.isfile(file_path):
        st.error(f"Error: The file '{file_path}' was not found.")
        st.stop()
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            st.error(f"Missing columns {missing_cols} in sheet '{sheet_name}' of '{file_path}'.")
            st.stop()
        df['Car'] = df['Car'].str.strip()
        return df
    except Exception as e:
        st.error(f"An error occurred while reading '{file_path}': {e}")
        st.stop()

# === Load Data ===
base_dir = os.path.dirname(os.path.abspath(__file__))
operating_costs_file = os.path.join(base_dir, 'data', 'challenger_operating_costs.xlsx')
fuel_economy_file = os.path.join(base_dir, 'data', 'challenger_fuel_economy.xlsx')
buying_prices_file = os.path.join(base_dir, 'data', 'challenger_buying_prices.xlsx')

df_operating_costs = load_excel_data(operating_costs_file, 'OperatingCosts', ['Car', 'Year1', 'Year2', 'Year3', 'Year4', 'Year5'])
df_fuel_economy = load_excel_data(fuel_economy_file, 'FuelEconomy', ['Car', 'FuelEconomy'])
df_buying_prices = load_excel_data(buying_prices_file, 'BuyingPrices', ['Car', 'BuyingPrice'])

challenger_operating_costs_data = df_operating_costs.set_index('Car').T.to_dict('list')
fuel_economy_data = pd.Series(df_fuel_economy.FuelEconomy.values, index=df_fuel_economy.Car).to_dict()
challenger_buying_prices = pd.Series(df_buying_prices.BuyingPrice.values, index=df_buying_prices.Car).to_dict()

# Predefined Scores
reliability_scores = {'Japanese': 0.9, 'Korean': 0.8, 'American': 0.8, 'German': 0.7}
safety_scores = {'Japanese': 0.7, 'Korean': 0.8, 'American': 0.85, 'German': 0.9}
challenger_nationality = {
    'Ford Explorer': 'American',
    'Honda Accord': 'Japanese',
    'Nissan Altima': 'Japanese',
    'Toyota Camry': 'Japanese',
    'Hyundai Sonata': 'Korean',
    'BMW 5 Series': 'German'
}
depreciation_rates = {
    'American': 0.15,
    'Japanese': 0.11,
    'German': 0.18,
    'Korean': 0.13
}

# Apply CSS for layout
st.markdown(
    """
    <style>
    .main {
        max-width: 70%;
        margin: auto;
    }
    .uniform-font {
        font-size: 16px;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Session State
if "page" not in st.session_state:
    st.session_state.page = "input"
if "inputs" not in st.session_state:
    st.session_state.inputs = {}

# === Input Page ===
if st.session_state.page == "input":
    st.title("🚗 Car Replacement Decision Support System")
    st.write("Please fill in the details below to get a recommendation on whether to keep your car or replace it.")

    with st.container():
        st.subheader("📊 Current Car Details")
        defender_market_value = st.number_input("Enter current car market value (USD):", min_value=0.0, value=11000.0, step=100.0)
        defender_operating_costs = []
        for i in range(1, 4):
            cost = st.number_input(f"Operating Cost for Year {i} (USD):", min_value=0.0, value=1000.0, step=50.0)
            defender_operating_costs.append(cost)

        defender_fuel_economy_kmpl = st.number_input("Current Car Fuel Economy (km/l):", min_value=0.0, value=12.7, step=0.1)
        style_score_defender_input = st.slider("Style Score for Current Car (0-10):", 0, 10, 6)
        origin_defender = st.selectbox("Select nationality:", ["Japanese", "Korean", "American", "German"])

        st.subheader("🚙 New Car Selection")
        challenger_car = st.selectbox("Select the new car you want to buy:", list(challenger_buying_prices.keys()))
        style_score_challenger_input = st.slider(f"Style Score for {challenger_car} (0-10):", 0, 10, 9)

        st.subheader("⚖️ Decision Factor Weights (0-10)")
        weights = {factor.lower().replace(" ", "_"): st.slider(f"Weight for {factor}:", 0, 10, 4 if factor in ['Style', 'Cost'] else 1) for factor in ['Style', 'Reliability', 'Cost', 'Fuel Economy', 'Safety']}

    if st.button("Get Recommendation"):
        st.session_state.inputs = {
            "defender_market_value": defender_market_value,
            "defender_operating_costs": defender_operating_costs,
            "defender_fuel_economy_kmpl": defender_fuel_economy_kmpl,
            "style_score_defender_input": style_score_defender_input,
            "origin_defender": origin_defender,
            "challenger_car": challenger_car,
            "style_score_challenger_input": style_score_challenger_input,
            "weights": weights,
        }
        st.session_state.page = "results"

# === Results Page ===
elif st.session_state.page == "results":
    inputs = st.session_state.inputs

    defender_market_value = inputs["defender_market_value"]
    defender_operating_costs = inputs["defender_operating_costs"]
    defender_fuel_economy_kmpl = inputs["defender_fuel_economy_kmpl"]
    style_score_defender_input = inputs["style_score_defender_input"]
    origin_defender = inputs["origin_defender"]
    challenger_car = inputs["challenger_car"]
    style_score_challenger_input = inputs["style_score_challenger_input"]
    weights = inputs["weights"]

    defender_fuel_economy_score = defender_fuel_economy_kmpl / 30
    style_score_defender = style_score_defender_input / 10
    style_score_challenger = style_score_challenger_input / 10

    predicted_defender_costs = predict_operating_costs(defender_operating_costs)
    dep_rate_defender = depreciation_rates.get(origin_defender, 0.1)
    dep_rate_challenger = depreciation_rates.get(challenger_nationality[challenger_car], 0.1)

    residual_value_defender = defender_market_value * (1 - dep_rate_defender) ** 5
    residual_value_challenger = challenger_buying_prices[challenger_car] * (1 - dep_rate_challenger) ** 5

    cash_flow_defender = [0] + [-cost for cost in predicted_defender_costs[:4]] + [-predicted_defender_costs[4] + residual_value_defender]
    cash_flow_challenger = [defender_market_value - challenger_buying_prices[challenger_car]] + [-cost for cost in challenger_operating_costs_data[challenger_car][:4]] + [-challenger_operating_costs_data[challenger_car][4] + residual_value_challenger]

    defender_pv = calculate_pv(cash_flow_defender)
    challenger_pv = calculate_pv(cash_flow_challenger)

    total_pv = defender_pv + challenger_pv
    cost_score_defender = challenger_pv / total_pv
    cost_score_challenger = defender_pv / total_pv

    defender_weighted_score = calculate_weighted_score(
        cost_score_defender, style_score_defender, reliability_scores[origin_defender],
        defender_fuel_economy_score, safety_scores[origin_defender], weights
    )
    challenger_weighted_score = calculate_weighted_score(
        cost_score_challenger, style_score_challenger, reliability_scores[challenger_nationality[challenger_car]],
        fuel_economy_data[challenger_car] / 30, safety_scores[challenger_nationality[challenger_car]], weights
    )

    st.title("📊 Results and Recommendation")
    st.markdown(f"""
    <div class="uniform-font">
    - Keeping your car will cost **${abs(defender_pv):,.2f}**.<br>
    - Replacing your car will cost **${abs(challenger_pv):,.2f}**.<br>
    - **Weighted Score: Current Car** = {defender_weighted_score:.4f}<br>
    - **Weighted Score: New Car** = {challenger_weighted_score:.4f}<br>
    - <b>Recommendation:</b> {'Replace the current car with the new one.' if challenger_weighted_score > defender_weighted_score else 'Keep your car.'}
    </div>
    """, unsafe_allow_html=True)

    st.subheader("💰 Cash Flow Diagrams")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Scenario 1: Keeping the Current Car**")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.bar(range(6), cash_flow_defender, color='blue')
        ax1.axhline(0, color='black', linewidth=0.8)
        ax1.set_ylim(min(cash_flow_defender) - 5000, max(cash_flow_defender) + 5000)
        ax1.set_title("Keeping the Current Car", fontsize=10)
        ax1.set_xlabel("Year", fontsize=8)
        ax1.set_ylabel("Cash Flow (USD)", fontsize=8)
        st.pyplot(fig1)

    with col2:
        st.markdown(f"**Scenario 2: Replacing with {challenger_car}**")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.bar(range(6), cash_flow_challenger, color='green')
        ax2.axhline(0, color='black', linewidth=0.8)
        ax2.set_ylim(min(cash_flow_challenger) - 5000, max(cash_flow_challenger) + 5000)
        ax2.set_title(f"Replacing with {challenger_car}", fontsize=10)
        ax2.set_xlabel("Year", fontsize=8)
        ax2.set_ylabel("Cash Flow (USD)", fontsize=8)
        st.pyplot(fig2)

    if st.button("Go Back"):
        st.session_state.page = "input"
