import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from io import BytesIO
from pathlib import Path

# Try to import Pillow; keep Image/ImageDraw/ImageFont as None if unavailable.
try:
    from PIL import Image as PILImage, ImageDraw as PILImageDraw, ImageFont as PILImageFont
    Image = PILImage
    ImageDraw = PILImageDraw
    ImageFont = PILImageFont
    _HAS_PIL = True
except Exception:
    Image = None
    ImageDraw = None
    ImageFont = None
    _HAS_PIL = False


# --- PAGE CONFIG ---
st.set_page_config(layout="centered", page_title="Interactive Macroeconomic Models")


# --- Utilities ---
def apply_layout_style(max_width: int = 1300, column_gap_px: int = 6):
    """Inject small CSS to constrain content width and tighten column gaps.
    This keeps charts visually consistent across large/small screens while
    still allowing responsive width scaling.
    """
    st.markdown(
        f"""
        <style>
        .block-container {{
            max-width: {max_width}px;
            padding-top: 0.75rem;
            padding-bottom: 2rem;
        }}
        [data-testid="column"] {{
            padding-left: {column_gap_px/2}px;
            padding-right: {column_gap_px/2}px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
def style_fig(fig, width: int = 800, height: int = 420):
    """Apply consistent sizing and margins to Plotly figures.
    Keeps a fixed height while letting width adapt to container.
    """
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=10, t=40, b=60),
    )
    return fig

def make_model_image(title: str, color: str = "#4c8bf5", size=(600, 360)):
    """Create a simple PNG image with the model title. Requires Pillow.

    Returns bytes suitable for st.image.
    """
    if not _HAS_PIL:
        return None
    img = Image.new("RGB", size, color)
    draw = ImageDraw.Draw(img)
    try:
        # Try common fonts; fallback to default if missing
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 36)
    except Exception:
        font = ImageFont.load_default()

    w, h = draw.textbbox((0, 0), title, font=font)[2:]
    draw.text(((size[0]-w)/2, (size[1]-h)/2), title, fill="white", font=font)
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


# --- MODEL & MATH FUNCTIONS ---


def simulate_ramsey(alpha, rho, theta, delta, n, g, k0, c0, T=100, dt=0.1):
    """Simulate a continuous-time Ramsey model with population (n) and technology (g).

    Variables k and c are per effective worker (divide by A*L). For Cobb-Douglas f(k)=k^alpha.

    Dynamics (per effective worker):
      k_dot = f(k) - c - (n + g + delta) * k
      c_dot = c * ( (f'(k) - (rho + theta * g + delta)) / theta )

    Discretized via forward Euler with step dt for illustration.
    """
    steps = int(T / dt)
    ks = np.zeros(steps + 1)
    cs = np.zeros(steps + 1)
    ts = np.linspace(0, T, steps + 1)
    ks[0] = k0
    cs[0] = c0
    for i in range(steps):
        k = ks[i]
        c = cs[i]
        f = k ** alpha
        fp = alpha * k ** (alpha - 1) if k > 0 else 0.0

        k_dot = f - c - (n + g + delta) * k
        c_dot = c * ((fp - (rho + theta * g + delta)) / theta)

        ks[i + 1] = max(0.0, k + dt * k_dot)
        cs[i + 1] = max(1e-8, c + dt * c_dot)

    return ts, ks, cs


# --- PAGES ---
def page_home():
    st.title("Interactive Macroeconomic Models")
    st.write("Choose a model below to open an interactive page.")

    # Create two columns for model cards
    c1, c2 = st.columns(2)

    # Solow card
    with c1:
        solow_path = Path("SolowImage.png")
        if solow_path.exists() and _HAS_PIL:
            try:
                img = Image.open(solow_path)
                st.image(img, use_container_width=True)
            except Exception:
                img_buf = make_model_image("Solow Growth Model", color="#4c8bf5")
                if img_buf is not None:
                    st.image(img_buf, use_container_width=True)
        else:
            img_buf = make_model_image("Solow Growth Model", color="#4c8bf5")
            if img_buf is not None:
                st.image(img_buf, use_container_width=True)

        st.caption("Classic exogenous growth model with capital accumulation.")
        if st.button("Open Solow"):
            st.session_state.page = "Solow"

    # Ramsey card
    with c2:
        ramsey_path = Path("RamseyImage.png")
        if ramsey_path.exists() and _HAS_PIL:
            try:
                img = Image.open(ramsey_path)
                st.image(img, use_container_width=True)
            except Exception:
                img_buf = make_model_image("Ramsey-Cass-Koopmans", color="#2ca02c")
                if img_buf is not None:
                    st.image(img_buf, use_container_width=True)
        else:
            img_buf = make_model_image("Ramsey-Cass-Koopmans", color="#2ca02c")
            if img_buf is not None:
                st.image(img_buf, use_container_width=True)

        st.caption("Optimal saving/consumption model with intertemporal choice.")
        if st.button("Open Ramsey"):
            st.session_state.page = "Ramsey"

    # If Pillow is not available, show textual fallbacks
    if not _HAS_PIL:
        st.info("Pillow not installed — using simple links instead of images.")
        if st.button("Go to Solow (text link)"):
            st.session_state.page = "Solow"
        if st.button("Go to Ramsey (text link)"):
            st.session_state.page = "Ramsey"

# --- Helper Functions for Styling ---
def format_number(num):
    return f"{num:,.2f}"

def format_small(num):
    return f"{num:,.3f}"

def page_solow():
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
            height=500,
            transition_duration=100
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
            height=500,
            transition_duration=500
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


def page_ramsey():
    st.title("Ramsey-Cass-Koopmans Model (Illustration)")
    st.write("This page simulates a stylized Ramsey model (continuous-time discretization).")

    st.sidebar.header("Ramsey Parameters")
    alpha = st.sidebar.slider("α (Output Elasticity)", 0.01, 1.0, 0.33, 0.01)
    rho = st.sidebar.slider("ρ (Rate of time preference)", 0.0, 1.0, 0.10, 0.001)
    theta = st.sidebar.slider("θ (Relative risk aversion)", 0.1, 5.0, 2.0, 0.1)
    delta = st.sidebar.slider("δ (Depreciation Rate)", 0.0, 0.5, 0.05, 0.001)
    n = st.sidebar.slider("n (Population growth)", 0.0, 0.1, 0.01, 0.001)
    g = st.sidebar.slider("g (Technology growth)", 0.0, 0.1, 0.02, 0.001)

    st.sidebar.divider()
    # Steady state in per effective worker variables:
    # f'(k*) = rho + theta*g + delta  =>  alpha * k*^{alpha-1} = rho + theta*g + delta
    if (rho + theta * g + delta) > 0 and alpha > 0:
        k_star = (alpha / (rho + theta * g + delta)) ** (1 / (1 - alpha)) if (1 - alpha) != 0 else np.nan
    else:
        k_star = np.nan
    # Resource at steady state per effective worker: c* = f(k*) - (n + g + delta) k*
    c_star = (k_star ** alpha - (n + g + delta) * k_star) if np.isfinite(k_star) else np.nan

    # Initial conditions: K0 at 90% of steady-state, A0=1, L0=1 => k0 = 0.9 * k*
    # default_k0 = float(0.9 * k_star) if np.isfinite(k_star) else 1.0
    # default_c0 = float(c_star) if np.isfinite(c_star) and c_star > 1e-8 else 1.0
    k0 = st.sidebar.number_input("Initial capital per effective worker k0", value=20.0, min_value=0.0)
    c0 = st.sidebar.number_input("Initial consumption per effective worker c0", value=1.5, min_value=1e-8)
    T = st.sidebar.number_input("Simulation horizon (T)", value=200, min_value=10)
    dt = st.sidebar.number_input("Time step (dt)", value=0.5, min_value=0.01)

    # Display options
    st.sidebar.subheader("Phase Diagram Options")
    show_arrows = st.sidebar.checkbox("Show trajectory arrows", value=True)
    show_fprime = st.sidebar.checkbox("Show f′(k) locus (dashed green)", value=True)
    show_saddle = st.sidebar.checkbox("Show saddle path (linearized)", value=True)

    # A0 and L0 per request
    A0 = 1.0
    L0 = 1.0

    ts, ks, cs = simulate_ramsey(alpha, rho, theta, delta, n, g, k0, c0, T=T, dt=dt)

    col1, col2, col3 = st.columns(3)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts, y=ks, name='k(t)', line=dict(color='royalblue')))
        if np.isfinite(k_star):
            fig.add_hline(y=k_star, line=dict(color='gray', dash='dash'), annotation_text='k*')
        fig.update_layout(title='Capital over time', xaxis_title='Time', yaxis_title='k')
        style_fig(fig)
        st.plotly_chart(fig, use_container_width=True, config={"responsive": True})

    with col2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=ts, y=cs, name='c(t)', line=dict(color='seagreen')))
        if np.isfinite(c_star):
            fig2.add_hline(y=c_star, line=dict(color='gray', dash='dash'), annotation_text='c*')
        fig2.update_layout(title='Consumption over time', xaxis_title='Time', yaxis_title='c')
        style_fig(fig2)
        st.plotly_chart(fig2, width=200, config={"responsive": True})

    with col3:
        st.header("Model Setup")
        st.write(f"Both the basic Solow model and Ramsey mmodel don't account for long-run growth without exogenous technological progress")
        st.write(f"While the production function satisfies neoclassical properties, it now include exogenous technological change:")
        st.latex(rf"Y(t) = F(K(t), A(t)L(t)),")
        st.write(f"where **A(t) = exp(gt)**")
        st.write(f"with **A(0) = {A0:.0f}** and where **g > 0** denotes the rate of technological progress.")
        st.write(f"Note that the production function imposes technological change through labour-augmenting (Harrod-neutral) technical change")
        st.write(f"Given constant returns to scale (CRS), we should normalise variables in terms of 'efficiency units of labour'")
        st.latex(rf"Y(t) = F(K(t), A(t)L(t)),")
        st.latex(r"""
        \hat{y}(t) \equiv \frac{Y(t)}{A(t)L(t)}
        = F\!\left(\frac{K(t)}{A(t)L(t)},\,1\right)
        \equiv f\big(k(t)\big),
        \quad\text{where}\quad
        k(t)\equiv\frac{K(t)}{A(t)L(t)}.
        """)
        st.write(f"We aren't interested in the behaviour of variables in efficiency units really, but rather in the evolution of per capita variables. However, rescaling the model this way provides convenience as variables in efficiency units do not grow in the long run.")
        st.subheader("Law of Motion of Capital")
        st.latex(r"""\hat{k}(t) = f\big(\hat{k}(t)\big) - \hat{c}(t) - \big(\delta + n + g)\hat{k}(t),""")
        st.subheader("Household Utility Function")
        st.latex(r"""U = \int_0^\infty \exp(-\rho - n - (1 - \theta)g)t) \frac{\hat{c}(t)^{1-\theta} - 1}{1 - \theta}dt""")


    # Phase diagram: consumption (c) vs capital (k) [per effective worker]
    # k̇ = 0 locus: c = f(k) - (n+g+δ) k
    # ċ = 0 locus: vertical at k* where f'(k*) = ρ + θ g + δ
    k_plot_max = max(np.nanmax(ks) * 1.1, (k_star * 1.2) if np.isfinite(k_star) else 30)
    k_vals = np.linspace(1e-6, max(0.1, k_plot_max), 400)
    f_vals = k_vals ** alpha
    c_kdot0 = f_vals - (n + g + delta) * k_vals
    fprime_vals = alpha * k_vals ** (alpha - 1)

    phase = go.Figure()
    phase.add_trace(go.Scatter(x=k_vals, y=c_kdot0, mode='lines',
                               name='k̇ = 0 (c = f(k) - δk)',
                               line=dict(color='orange', width=3)))

    # ċ = 0 vertical line at k*
    if np.isfinite(k_star):
        phase.add_vline(x=k_star, line=dict(color='royalblue', dash='dash'), annotation_text='ċ = 0', annotation_position='top left')

    # Optional: f′(k) locus (dashed green)
    if show_fprime:
        phase.add_trace(go.Scatter(x=k_vals, y=fprime_vals, mode='lines',
                                   name="f′(k)", line=dict(color='green', dash='dash', width=2)))

    # Trajectory in phase space
    phase.add_trace(go.Scatter(x=ks, y=cs, mode='lines+markers', name='Trajectory (k(t), c(t))',
                               line=dict(color='lime', width=2), marker=dict(size=4)))

    # Optional: red arrows to indicate direction
    if show_arrows and len(ks) > 2:
        step = max(2, len(ks) // 12)
        for i in range(step, len(ks), step):
            phase.add_annotation(x=ks[i], y=cs[i], ax=ks[i-1], ay=cs[i-1],
                                 xref='x', yref='y', axref='x', ayref='y',
                                 showarrow=True, arrowhead=3, arrowsize=3, arrowwidth=1.5, arrowcolor='red')

    # Start and steady-state markers
    phase.add_trace(go.Scatter(x=[ks[0]], y=[cs[0]], mode='markers', name='Start',
                               marker=dict(color='white', size=10, line=dict(color='black', width=1))))
    if np.isfinite(k_star) and np.isfinite(c_star):
        phase.add_trace(go.Scatter(x=[k_star], y=[c_star], mode='markers', name='Steady state',
                                   marker=dict(color='cyan', size=12, symbol='circle')))

    # Optional: analytical saddle path (linearized around steady state)
    if show_saddle and np.isfinite(k_star) and np.isfinite(c_star) and theta > 0:
        # Jacobian at steady state for continuous-time system
        fpp = alpha * (alpha - 1) * (k_star ** (alpha - 2)) if k_star > 0 else 0.0
        a11 = (alpha * k_star ** (alpha - 1)) - (n + g + delta)  # ∂k_dot/∂k
        a12 = -1.0                                              # ∂k_dot/∂c
        a21 = (c_star * fpp) / theta                            # ∂c_dot/∂k
        a22 = 0.0                                               # ∂c_dot/∂c at SS

        # Eigenvalues λ from |A - λI| = 0
        tr = a11 + a22
        det = a11 * a22 - a12 * a21
        disc = tr * tr - 4 * det
        if disc >= 0:
            lambda1 = 0.5 * (tr + np.sqrt(disc))
            lambda2 = 0.5 * (tr - np.sqrt(disc))
            # Stable eigenvalue: negative one
            lam_stable = lambda1 if lambda1 < lambda2 else lambda2
            # Eigenvector v satisfies (A - λI) v = 0 => choose v1=1 => v2 from first row
            v1 = 1.0
            v2 = -(a11 - lam_stable) * v1 / a12  # since a12 != 0
            m = v2 / v1  # slope dc/dk near SS
            k_line = np.linspace(max(1e-6, k_vals.min()), k_vals.max(), 200)
            c_line = c_star + m * (k_line - k_star)
            phase.add_trace(go.Scatter(x=k_line, y=c_line, mode='lines', name='Saddle path (linearized)',
                                       line=dict(color='magenta', dash='dot', width=2)))

            # Annotate the saddle path with directional arrows pointing toward the steady state
            # Use a subset of points; for each point, draw an arrow head closer to (k*, c*)
            if len(k_line) > 2:
                idxs = np.linspace(0, len(k_line) - 1, num=10, dtype=int)
                for j in idxs:
                    ki = float(k_line[j])
                    ci = float(c_line[j])
                    if not (np.isfinite(ki) and np.isfinite(ci)):
                        continue
                    # Only add arrows when c(t) != 1; condition matches the request
                    if abs(ci - 1.0) < 1e-6:
                        continue
                    # Vector pointing toward steady state
                    vx = k_star - ki
                    vy = c_star - ci
                    norm = (vx**2 + vy**2) ** 0.5
                    if norm < 1e-9:
                        continue
                    frac = 0.2  # arrow length as a fraction toward the steady state
                    hx = ki + frac * vx
                    hy = ci + frac * vy
                    phase.add_annotation(
                        x=hx, y=hy, ax=ki, ay=ci,
                        xref='x', yref='y', axref='x', ayref='y',
                        showarrow=True, arrowhead=3, arrowsize=1.5, arrowwidth=2, arrowcolor='red'
                    )

            # Label the saddle path near the steady state
            phase.add_annotation(x=k_star, y=c_star, text='Saddle path', showarrow=False,
                                 xshift=12, yshift=10, font=dict(color='red'))

    # Axis limits: default to at least x<=30 and y<=3.0, but expand if data exceed them
    x_axis_max = max(k_vals.max() * 1.02, 30.0)
    y_axis_max = max(max(c_kdot0.max(), np.nanmax(cs)) * 1.1, 3.0)

    phase.update_layout(
        title='Phase Diagram: Consumption vs Capital',
        xaxis_title='Capital (k)',
        yaxis_title='Consumption (c)',
        xaxis=dict(range=[0, x_axis_max]),
        yaxis=dict(range=[0, y_axis_max])
    )

    style_fig(phase)
    st.plotly_chart(phase, theme="streamlit", config={"responsive": True})

    st.subheader('Steady state (analytical)')
    st.write(f'k* ≈ {k_star:.3f} , c* ≈ {c_star:.3f}')


# --- APP ROUTING ---
def main():
    # Apply global layout styling (consistent width and gaps)
    apply_layout_style()
    # Initialize session state for page selection
    if 'page' not in st.session_state:
        st.session_state.page = 'Home'

    # Sidebar navigation
    page = st.sidebar.radio('Page', ['Home', 'Solow', 'Ramsey'], index=['Home', 'Solow', 'Ramsey'].index(st.session_state.page))
    st.session_state.page = page

    # Render chosen page
    if st.session_state.page == 'Home':
        page_home()
    elif st.session_state.page == 'Solow':
        page_solow()
    elif st.session_state.page == 'Ramsey':
        page_ramsey()


if __name__ == '__main__':
    main()