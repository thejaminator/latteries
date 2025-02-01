import plotly.io as pio


import plotly.express as px
import pandas as pd

"""
model	win rate
ITC: DeepSeek R1	66.9% (± 7,2%)
ITC: Gemini	62.7% (± 4.7%)
ITC: Qwen	71.2% (± 5.1%)
"""
data = [
    {
        "Model": "Gemini",
        "Win rate": 62.7,
        "Error": 4.7,
    },
    {
        "Model": "Qwen",
        "Win rate": 71.2,
        "Error": 5.1,
    },
    {
        "Model": "Deepseek",
        "Win rate": 66.9,
        "Error": 7.2,
    },
]
df = pd.DataFrame(data)

# Create bar plot with error bars
fig = px.bar(
    df,
    x="Model",
    y="Win rate",
    error_y="Error",
    hover_data=["Model"],
    text="Win rate",
    color_discrete_sequence=["#46d09c"],  # Set the color to green
)

# add text labels by model
# fig.update_traces(textposition='outside', texttemplate='%{text:.1f}%', text=df['Articulation Rate'])
padd = "&nbsp;" * 8
trace_font = dict(size=12)
# don't show decimal places
fig.update_traces(textposition="outside", texttemplate=padd + "%{text:.0f}%", textfont_size=16, textangle=0)

# Add red dotted line at 50%
fig.add_shape(
    type="line",
    x0=-0.5,
    x1=len(df) - 0.5,
    y0=50,
    y1=50,
    line=dict(color="red", width=2, dash="dot"),
)

# Combine update layout
fig.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white",
    xaxis=dict(
        showline=True,
        linewidth=1,
        linecolor="black",
    ),
    yaxis=dict(
        showline=True,
        linewidth=1,
        linecolor="black",
    ),
    xaxis2=dict(
        ticktext=df["Model"],
        tickvals=list(range(len(df))),
        overlaying="x",
        side="bottom",
        title="Model",
        anchor="y",
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
    ),
    xaxis_title=None,
    yaxis_title="",
    width=400,
    height=400,
    showlegend=False,
    font=dict(size=18),  # Set font size to 14
    # tickfont=dict(size=16),
    titlefont=dict(size=16),
)

fig.show()
# hack to fix mathjax

pio.kaleido.scope.mathjax = None
pdf_name = "win_rates_gpt4o.pdf"
# remove margins
fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
fig.write_image(pdf_name)
