import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.integrate import odeint
from scipy.linalg import eig

# --- Page Configuration ---
st.set_page_config(
    page_title="Interactive Ramsey-Cass-Koopmans Model",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions ---
def format_number(num):
    return f"{num:,.2f}"

def format_small(num):
    return f"{num:,.3f}"

# --- Sidebar: Parameters ---
st.sidebar.header("⚙️ Model Parameters")

# Reset Logic
if 'reset_ramsey' not in st.session_state:
    st.session_state.reset_ramsey = False

def reset_defaults():
    st.session_state.alpha_r = 0.33
    st.session_state.delta_r = 0.06
    st.session_state.n_r = 0.01
    st.session_state.g_r = 0.02
    st.session_state.rho_r = 0.04
    st.session_state.gamma_r = 2.00
    st.session_state.k_curr_r = 20.0

if st.sidebar.button("🔄 Reset to Defaults"):
    reset_defaults()

# Sliders
alpha = st.sidebar.slider("Capital Share (α)", 0.1, 0.9, 0.33, 0.01, key='alpha_r', help="Elasticity of output with respect to capital.")
delta = st.sidebar.slider("Depreciation (δ)", 0.0, 0.2, 0.06, 0.005, key='delta_r', help="Rate of capital depreciation.")
n = st.sidebar.slider("Pop. Growth (n)", 0.0, 0.1, 0.01, 0.005, key='n_r', help="Rate of population growth.")
g = st.sidebar.slider("Tech. Growth (g)", 0.0, 0.1, 0.02, 0.005, key='g_r', help="Rate of labor-augmenting technological change.")
rho = st.sidebar.slider("Discount Rate (ρ)", 0.0, 0.2, 0.04, 0.005, key='rho_r', help="Time preference rate.")
gamma = st.sidebar.slider("Inv. Elast. subst. (γ/θ)", 0.1, 10.0, 2.00, 0.1, key='gamma_r', help="Inverse elasticity of intertemporal substitution (CRRA coefficient).")

st.sidebar.markdown("---")
k_current = st.sidebar.slider("Current Capital (k_0)", 1.0, 50.0, 20.0, 0.5, key='k_curr_r')

# --- Model Logic & Dynamics ---

# 1. Calculate Steady State
# Euler Relation: f'(k) = rho + delta + gamma * g
# alpha * k^(alpha-1) = rho + delta + gamma * g
# k^(alpha-1) = (rho + delta + gamma * g) / alpha
# k* = [ (rho + delta + gamma * g) / alpha ] ^ (1 / (alpha - 1))
denom = rho + delta + (gamma * g)
k_star = (denom / alpha) ** (1 / (alpha - 1))

# Steady State Consumption from k_dot = 0
# c* = f(k*) - (n + g + delta)k*
y_star = k_star ** alpha
break_even_investment = (n + g + delta) * k_star
c_star = y_star - break_even_investment

# 2. Phase Diagram Loci
k_max = max(k_star * 2, 60)
k_vals = np.linspace(0.1, k_max, 300)

# k_dot = 0 Locus: c = k^alpha - (n + g + delta)k
# This is the "Solow" steady state curve (inverted U)
c_k_locus = (k_vals ** alpha) - ((n + g + delta) * k_vals)
# Clip negatives for plotting cleanliness
c_k_locus = np.maximum(c_k_locus, 0)

# c_dot = 0 Locus: k = k_star (Vertical line)
# We just plot a vertical line at k_star later.

# 3. Saddle Path Calculation (Linearization & Backwards Integration)
# We linearize around (k*, c*) to find the stable eigenvector.
# Jacobian J:
# J11 = d(k_dot)/dk = alpha*k^(alpha-1) - (n+g+delta) = (rho+delta+gamma*g) - (n+g+delta) = rho - n + g(gamma-1)
J11 = rho - n + g * (gamma - 1)
# J12 = d(k_dot)/dc = -1
J12 = -1.0
# J21 = d(c_dot)/dk = (c/gamma) * f''(k)
f_double_prime = alpha * (alpha - 1) * (k_star ** (alpha - 2))
J21 = (c_star / gamma) * f_double_prime
# J22 = d(c_dot)/dc = 0 (since f'(k) - ... = 0 at steady state)
J22 = 0.0

J = np.array([[J11, J12], [J21, J22]])
vals, vecs = eig(J)

# Find the negative eigenvalue (stable arm)
# We expect one positive and one negative eigenvalue for saddle path stability.
neg_eig_idx = np.argmin(vals.real) 
stable_vec = vecs[:, neg_eig_idx].real

# Slope of the saddle path at steady state
slope = stable_vec[1] / stable_vec[0]

# Define the system for odeint (Time Reversal)
# We integrate BACKWARDS from near the steady state to trace the saddle path.
def ramsey_system_reverse(state, t):
    k, c = state
    # Protect against negative values during integration
    if k <= 0 or c <= 0: return [0, 0]
    
    dk_dt = (k ** alpha) - c - (n + g + delta) * k
    
    mpk = alpha * (k ** (alpha - 1))
    dc_dt = (c / gamma) * (mpk - (rho + delta + gamma * g))
    
    # Return NEGATIVE derivatives to go backwards in time
    return [-dk_dt, -dc_dt]

# Integrate backwards from k* +/- epsilon
epsilon = 0.01
# Path 1: Go left (towards origin)
start_left = [k_star - epsilon, c_star - slope * epsilon]
t_span = np.linspace(0, 100, 500)
path_left = odeint(ramsey_system_reverse, start_left, t_span)

# Path 2: Go right (away from origin)
start_right = [k_star + epsilon, c_star + slope * epsilon]
path_right = odeint(ramsey_system_reverse, start_right, t_span)

# Combine, filter, and sort for plotting
saddle_k = np.concatenate([path_left[:, 0][::-1], path_right[:, 0]])
saddle_c = np.concatenate([path_left[:, 1][::-1], path_right[:, 1]])

# Filter valid range
mask = (saddle_k > 0) & (saddle_k < k_max) & (saddle_c > 0)
saddle_k = saddle_k[mask]
saddle_c = saddle_c[mask]

# Interpolate to find c_current for the selected k_current
if k_current > np.min(saddle_k) and k_current < np.max(saddle_k):
    c_current = np.interp(k_current, saddle_k, saddle_c)
else:
    c_current = 0 # Should not happen usually in reasonable ranges

# --- Main Layout ---

st.title("📉 Interactive Ramsey-Cass-Koopmans Model")
st.markdown("Explore the optimization dynamics, saddle path stability, and the modified golden rule.")

col_charts, col_math = st.columns([2, 1], gap="large")

with col_charts:
    fig = go.Figure()

    # 1. k_dot = 0 Locus (Orange)
    fig.add_trace(go.Scatter(
        x=k_vals, y=c_k_locus,
        mode='lines', name='k̇ = 0 Locus',
        line=dict(color='#f97316', width=3)
    ))
    # Label for k_dot
    fig.add_annotation(
        x=k_vals[-10], y=c_k_locus[-10],
        text="k̇ = 0", showarrow=False, font=dict(color='#f97316')
    )

    # 2. c_dot = 0 Locus (Blue Vertical)
    fig.add_vline(x=k_star, line_width=3, line_color='#3b82f6', name='ċ = 0 Locus')
    fig.add_annotation(
        x=k_star, y=max(c_k_locus)*1.1,
        text="ċ = 0", showarrow=False, font=dict(color='#3b82f6')
    )

    # 3. Saddle Path (Green Dashed)
    fig.add_trace(go.Scatter(
        x=saddle_k, y=saddle_c,
        mode='lines', name='Saddle Path',
        line=dict(color='#4ade80', width=2, dash='dash')
    ))

    # 4. Current Point O (User Interactive)
    fig.add_trace(go.Scatter(
        x=[k_current], y=[c_current],
        mode='markers', name='Current State (O)',
        marker=dict(color='#a3e635', size=15, line=dict(color='white', width=2))
    ))
    
    # 5. Steady State Point
    fig.add_trace(go.Scatter(
        x=[k_star], y=[c_star],
        mode='markers', name='Steady State',
        marker=dict(color='white', size=8)
    ))

    # Layout styling
    fig.update_layout(
        title="Phase Diagram (c vs k)",
        xaxis_title="Capital Stock (k)",
        yaxis_title="Consumption (c)",
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        margin=dict(l=20, r=20, t=40, b=20),
        height=550,
        yaxis=dict(range=[0, max(c_k_locus)*1.2]),
        xaxis=dict(range=[0, k_max])
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.caption("The **Green Dashed Line** represents the Saddle Path (Stable Arm). Optimal households will always choose consumption to land exactly on this path to converge to the Steady State.")

with col_math:
    st.subheader("🧮 Math Engine")
    
    # Style container
    container_css = """
    <style>
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
    }
    </style>
    """
    st.markdown(container_css, unsafe_allow_html=True)

    # 1. Steady State Capital (c_dot = 0 condition)
    # Note: The image in prompt had a specific layout for this, I follow that structure.
    st.markdown(f"""
    <div class="metric-card">
        <div style="color: #3b82f6; font-weight: bold; margin-bottom: 5px;">Steady-state consumption (ċ = 0)</div>
        <div style="font-size: 0.9em; margin-bottom: 5px;">Modified Golden Rule:</div>
        <div style="text-align: center; margin-bottom: 5px;">
             $\\alpha k^{{\\alpha-1}} = \\rho + \\delta + \\gamma g$
        </div>
        <div style="font-size: 0.85em; color: #94a3b8; text-align: center;">
            ${alpha} k^{{{alpha-1:.2f}}} = {rho} + {delta} + {gamma}({g})$
        </div>
        <div style="text-align: center; font-weight: bold; font-size: 1.2em; color: #f8fafc; margin-top: 5px;">
            $k^* = {format_number(k_star)}$
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Steady State Consumption (k_dot = 0 condition)
    # In SS, c = f(k) - (n+g+d)k
    st.markdown(f"""
    <div class="metric-card">
        <div style="color: #f97316; font-weight: bold; margin-bottom: 5px;">Steady-state capital (k̇ = 0)</div>
        <div style="font-size: 0.9em; margin-bottom: 5px;">Resource Constraint:</div>
        <div style="text-align: center; margin-bottom: 5px;">
             $c^* = (k^*)^\\alpha - (n + g + \\delta)k^*$
        </div>
        <div style="font-size: 0.85em; color: #94a3b8; text-align: center;">
            $c^* = {format_number(k_star)}^{{{alpha}}} - ({n+g+delta:.3f}){format_number(k_star)}$
        </div>
        <div style="text-align: center; font-weight: bold; font-size: 1.2em; color: #f8fafc; margin-top: 5px;">
            $c^* = {format_number(c_star)}$
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 3. Current State Analysis
    st.markdown(f"""
    <div class="metric-card">
        <div style="color: #a3e635; font-weight: bold; margin-bottom: 5px;">Current State (Point O)</div>
        <div>Given $k_0 = {k_current}$:</div>
        <div>Optimal $c_0$ on Saddle Path:</div>
        <div style="font-size: 1.5em; font-weight: bold; color: white;">{format_number(c_current)}</div>
    </div>
    """, unsafe_allow_html=True)

    # Check for Golden Rule
    # GR is where MPK = n + g + delta (Consumption maximized)
    # In Ramsey, MPK = rho + delta + gamma*g (Utility maximized)
    # If rho = n and gamma = 1 (and g=0 or accounted for), they align.
    mpk_star = alpha * (k_star ** (alpha - 1))
    st.info(f"**Steady State Interest Rate (MPK - δ):**\n\n$r^* = {(mpk_star - delta):.4f}$")