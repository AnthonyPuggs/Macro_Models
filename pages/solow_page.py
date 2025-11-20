import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="Interactive Solow Model",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions for Styling ---
def format_number(num):
    return f"{num:,.2f}"

# --- Sidebar: Parameters ---
st.sidebar.header("⚙️ Model Parameters")

# Reset Logic
if 'reset_trigger' not in st.session_state:
    st.session_state.reset_trigger = False

def reset_defaults():
    st.session_state.alpha = 0.50
    st.session_state.s = 0.20
    st.session_state.delta = 0.06
    st.session_state.k_current = 50.0

if st.sidebar.button("🔄 Reset to Defaults"):
    reset_defaults()

# Sliders with session state keys for reset capability
alpha = st.sidebar.slider(
    "Output Elasticity (α)", 
    min_value=0.1, max_value=0.9, value=0.50, step=0.01, key='alpha',
    help="Returns to capital. Higher α means capital is more productive."
)

s = st.sidebar.slider(
    "Savings Rate (s)", 
    min_value=0.01, max_value=0.90, value=0.20, step=0.01, key='s',
    help="Fraction of output invested."
)

delta = st.sidebar.slider(
    "Depreciation Rate (δ)", 
    min_value=0.01, max_value=0.20, value=0.06, step=0.005, key='delta',
    help="Rate at which capital wears out."
)

st.sidebar.markdown("---")

k_current = st.sidebar.slider(
    "Current Capital (k_t)", 
    min_value=1.0, max_value=200.0, value=50.0, step=1.0, key='k_current',
    help="Starting capital stock."
)

# --- Calculations ---

# 1. Steady State
# s * k^alpha = delta * k  =>  k* = (s/delta)^(1/(1-alpha))
k_star = (s / delta) ** (1 / (1 - alpha))
y_star = k_star ** alpha
golden_rule_s = alpha

# 2. Current State Dynamics
y_current = k_current ** alpha
investment_current = s * y_current
depreciation_current = delta * k_current
change_in_k = investment_current - depreciation_current
k_next = k_current + change_in_k

# --- Main Layout ---

st.title("📈 Interactive Solow Growth Model")
st.markdown("Simulate capital accumulation and steady-state convergence.")

# Create two columns: Left for Charts, Right for Math
col_charts, col_math = st.columns([2, 1], gap="large")

with col_charts:
    # Tabs for Phase Diagram vs Time Series
    tab1, tab2 = st.tabs(["Phase Diagram", "Convergence over Time"])
    
    with tab1:
        # --- Phase Diagram Data Generation ---
        max_k_plot = max(k_current, k_star) * 1.5
        k_values = np.linspace(0, max_k_plot, 200)
        
        # Avoid divide by zero or log(0) issues
        k_values[0] = 0.01
        
        y_values = k_values ** alpha
        i_values = s * y_values
        dep_values = delta * k_values
        
        # --- Phase Diagram Plot ---
        fig_phase = go.Figure()

        # Output Curve
        fig_phase.add_trace(go.Scatter(
            x=k_values, y=y_values, mode='lines', name='Output f(k)',
            line=dict(color='#3b82f6', width=3)
        ))

        # Investment Curve
        fig_phase.add_trace(go.Scatter(
            x=k_values, y=i_values, mode='lines', name='Investment s·f(k)',
            line=dict(color='#10b981', width=3)
        ))

        # Depreciation Line
        fig_phase.add_trace(go.Scatter(
            x=k_values, y=dep_values, mode='lines', name='Depreciation δ·k',
            line=dict(color='#ef4444', width=3)
        ))

        # Current k Line
        fig_phase.add_vline(x=k_current, line_width=2, line_dash="dash", line_color="orange", annotation_text="Current k", annotation_position="top right")
        
        # Steady State Line
        fig_phase.add_vline(x=k_star, line_width=2, line_dash="dash", line_color="purple", annotation_text="Steady State k*", annotation_position="bottom right")

        fig_phase.update_layout(
            title="Phase Diagram: Capital Accumulation",
            xaxis_title="Capital Stock (k)",
            yaxis_title="Output / Investment",
            template="plotly_dark",
            hovermode="x unified",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            margin=dict(l=20, r=20, t=40, b=20),
            height=500
        )
        st.plotly_chart(fig_phase, use_container_width=True)

    with tab2:
        # --- Time Series Data Generation ---
        periods = 50
        ts_k = [k_current]
        ts_y = []
        ts_c = [] # Consumption
        
        curr_k = k_current
        for _ in range(periods + 1):
            curr_y = curr_k ** alpha
            curr_i = s * curr_y
            curr_dep = delta * curr_k
            curr_c = curr_y - curr_i
            
            ts_y.append(curr_y)
            ts_c.append(curr_c)
            
            # Update for next loop (store next k)
            next_k = curr_k + curr_i - curr_dep
            ts_k.append(next_k)
            curr_k = next_k
        
        # Remove the extra k from the end for plotting consistency
        ts_k = ts_k[:-1] 
        
        df_ts = pd.DataFrame({
            'Period': range(periods + 1),
            'Output': ts_y,
            'Consumption': ts_c,
            'Capital': ts_k
        })

        # --- Time Series Plot ---
        fig_ts = go.Figure()

        fig_ts.add_trace(go.Scatter(
            x=df_ts['Period'], y=df_ts['Output'], fill='tozeroy', mode='lines', name='Output (y)',
            line=dict(color='#3b82f6')
        ))
        
        fig_ts.add_trace(go.Scatter(
            x=df_ts['Period'], y=df_ts['Consumption'], fill='tozeroy', mode='lines', name='Consumption (c)',
            line=dict(color='#10b981')
        ))

        fig_ts.add_hline(y=y_star, line_dash="dot", line_color="purple", annotation_text="Steady State Output")

        fig_ts.update_layout(
            title="Convergence over Time",
            xaxis_title="Time Period (t)",
            yaxis_title="Level",
            template="plotly_dark",
            hovermode="x unified",
            margin=dict(l=20, r=20, t=40, b=20),
            height=500
        )
        st.plotly_chart(fig_ts, use_container_width=True)

    # Dynamic Text Interpretation
    steady_state_status = "growing" if change_in_k > 0 else "shrinking"
    color_status = "green" if change_in_k > 0 else "red"
    
    st.caption(f"""
    **Analysis:** The economy is converging to a steady state capital of **{format_number(k_star)}**. 
    Currently at **k = {k_current}**, the capital stock is :{color_status}[{steady_state_status}].
    """)

with col_math:
    st.subheader("🧮 Math Engine")
    
    # Container style
    container_css = """
    <style>
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #f8fafc;
    }
    </style>
    """
    st.markdown(container_css, unsafe_allow_html=True)

    # 1. Output
    st.info(f"**1. Output (y)**\n\n$y_t = k_t^\\alpha = {k_current}^{{{alpha}}} = {format_number(y_current)}$")

    # 2. Savings
    st.success(f"**2. Savings (i)**\n\n$i_t = s \\times y_t = {s} \\times {format_number(y_current)} = {format_number(investment_current)}$")

    # 3. Depreciation
    st.error(f"**3. Depreciation (δk)**\n\n$\\delta k_t = {delta} \\times {k_current} = {format_number(depreciation_current)}$")

    # 4. Accumulation
    change_sign = "+" if change_in_k > 0 else ""
    st.warning(f"""
    **4. Next Period (k+1)**
    
    $k_{{t+1}} = k_t + i_t - \\delta k_t$
    
    $k_{{t+1}} = {k_current} + {format_number(investment_current)} - {format_number(depreciation_current)}$
    
    **$k_{{t+1}} = {format_number(k_next)}$**
    
    (Net Change: {change_sign}{format_number(change_in_k)})
    """)

    # Golden Rule Check
    if abs(s - golden_rule_s) < 0.02:
        st.toast("🌟 You are near the Golden Rule Savings Rate!", icon="🏆")
    
    st.markdown("---")
    st.markdown(f"**Golden Rule Savings Rate:** $s_{{gold}} = \\alpha = {alpha:.2f}$")