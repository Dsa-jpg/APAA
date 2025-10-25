import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from data.data_generator import DataGenerator
from neuron.linear_neuron import LinearNeuron

st.set_page_config(page_title="Adaptive Processing Algorithms", layout="wide")

st.sidebar.header("⚙Signal parameters")
a1 = st.sidebar.slider("a1 (Amplitude 1)", 10, 200, 100, step=5)
a2 = st.sidebar.slider("a2 (Amplitude 2)", 10, 200, 20, step=5)
f1 = st.sidebar.slider("f1 (Frequency 1)", 50, 500, 250, step=10)
f2 = st.sidebar.slider("f2 (Frequency 2)", 50, 500, 100, step=10)

st.sidebar.header("LNU parameters")
learning_rate = st.sidebar.number_input(
    "Learning rate", min_value=0.0001, max_value=1.000, value=0.001, step=0.0001, format="%.4f"
)
epochs = st.sidebar.slider("Epochs", 10, 500, 100, step=10)
look_back_window = st.sidebar.slider("Look-back window", 1, 10, 5, step=1)
future_steps = st.sidebar.slider("Future steps", 1, 200, 10, step=1)

run_simulation = st.sidebar.button("Run simulation")
tabs = st.tabs(["Signal", "LNU"])


def plot_signal(t, signal):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=signal, mode='lines', name='Signal', line=dict(color='blue')))
    fig.update_layout(
        title="Generated Signal",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        height=500,
        width=1200,
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_LNU(t, normalized_signal, predicted_values, future_pred, Error, weights_history):
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.15,
        subplot_titles=("Normalized Signal + Predictions", "Error Progress", "Weights History"),
        row_heights=[0.4, 0.4, 0.4]
    )

    fig.add_trace(
        go.Scatter(x=t[:len(normalized_signal)], y=normalized_signal, mode='lines', name='Normalized Signal',
                   line=dict(color='red')),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=t[:len(predicted_values)], y=predicted_values, mode='markers', name='Predicted Values',
                   line=dict(color='green')),
        row=1, col=1
    )

    if future_pred is not None:
        fig.add_trace(
            go.Scatter(
                x=t[len(predicted_values):],
                y=future_pred,
                mode='lines+markers',
                name='Future Prediction',
                line=dict(color='orange', dash='dot'),
                marker=dict(symbol='circle-open')
            ),
            row=1, col=1
        )

    fig.add_trace(
        go.Scatter(x=list(range(1, len(Error) + 1)), y=Error, mode='lines', name='Error', line=dict(color='blue')),
        row=2, col=1
    )

    for i in range(weights_history.shape[1]):
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(weights_history) + 1)),
                y=weights_history[:, i],
                mode='lines',
                name=f'Weight {i + 1}'
            ),
            row=3, col=1
        )

    fig.update_layout(
        height=900,
        width=1100,
        showlegend=True,
        title_text="Linear Neuron Learning Visualization",
        template="plotly_white",
        margin=dict(l=50, r=50, t=80, b=50),
        font=dict(size=12)
    )

    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=2, col=1)
    fig.update_xaxes(title_text="Epoch", row=3, col=1)

    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_yaxes(title_text="%", row=2, col=1)
    fig.update_yaxes(title_text="Weight value", row=3, col=1)

    st.plotly_chart(fig, use_container_width=True)


with tabs[0]:
    st.header("Generated Signal")
    if run_simulation:
        data = DataGenerator(a1=a1, a2=a2, f1=f1, f2=f2)
        t, signal = data.generate()
        plot_signal(t, signal)
    else:
        st.info("Click run simulation.")

with tabs[1]:
    st.header("Linear Neuron Learning (LNU)")
    if run_simulation:
        data = DataGenerator(a1=a1, a2=a2, f1=f1, f2=f2)
        t, signal = data.generate()
        normalized_signal = data.normalize(signal)

        LNU = LinearNeuron(learning_rate=learning_rate, signal=normalized_signal, t=t,
                           look_back_window=look_back_window)
        LNU.learn(epochs=epochs)

        predicted_values = [np.nan] * look_back_window + list(LNU.pred.values())

        future_pred = LNU.predict_future(future_steps)
        extended_t = np.arange(len(normalized_signal) + future_steps)

        plot_LNU(extended_t, normalized_signal, predicted_values, future_pred, LNU.error, np.array(LNU.weights_history))
    else:
        st.info("Click run simulation.")
