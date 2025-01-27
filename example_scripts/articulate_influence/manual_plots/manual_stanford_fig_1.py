import plotly.io as pio


import plotly.express as px
import pandas as pd

# Create dataframe
# mock data
alibaba_group = "Qwen"
google_group = "Gemini"
deepseek_group = "Deepseek"

"""
model	Professor
ITC: DeepSeek R1	59.4% (± 6.4%)
ITC: Gemini	68.2% (± 9.8%)
ITC: Qwen	46.9% (± 7.3%)
Claude-3.5-Sonnet	6.7% (± 5.2%)
Deepseek-Chat-v3	6.5% (± 3.6%)
GPT-4o	2.4% (± 3.3%)
Gemini-2.0-Flash-Exp	13.0% (± 6.2%)
Grok-2-1212	4.9% (± 4.2%)
Qwen-72b-Instruct	5.3% (± 3.6%)
"""
data = [
    {
        "Model": "Qwen 72b",
        "Articulation Rate": 5.3,
        "Type": "Non-ITC",
        "Group": alibaba_group,
        "Error": 3.6,
    },
    {
        "Model": "QwQ",
        "Articulation Rate": 46.9,
        "Type": "Inference-Time-Compute (ITC)",
        "Group": alibaba_group,
        "Error": 7.3,
    },
    {
        "Model": "Gemini",
        "Articulation Rate": 13.0,
        "Type": "Non-ITC",
        "Group": google_group,
        "Error": 6.2,
    },
    {
        "Model": "Gemini-Thinking",
        "Articulation Rate": 68.2,
        "Type": "Inference-Time-Compute (ITC)",
        "Group": google_group,
        "Error": 9.8,
    },
    {
        "Model": "Deepseek-R1",
        "Articulation Rate": 59.4,
        "Type": "Inference-Time-Compute (ITC)",
        "Group": deepseek_group,
        "Error": 6.4,
    },
    {
        "Model": "Deepseek-V3",
        "Articulation Rate": 6.5,
        "Type": "Non-ITC",
        "Group": deepseek_group,
        "Error": 3.6,
    },
]
df = pd.DataFrame(data)

# Create bar plot with error bars
fig = px.bar(
    df,
    x="Group",
    y="Articulation Rate",
    color="Type",
    error_y="Error",
    barmode="group",
    #  title='Articulation Rate by Model Type',
    hover_data=["Model"],
    text="Articulation Rate",
)

# add text labels by model
# fig.update_traces(textposition='outside', texttemplate='%{text:.1f}%', text=df['Articulation Rate'])
padd = "&nbsp;" * 8
trace_font = dict(size=12)
# don't show decimal places
fig.update_traces(textposition="outside", texttemplate=padd + "%{text:.0f}%", textfont=trace_font)


# Set white background and add spines
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
)

# Add secondary x-axis with model names
fig.update_layout(
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
    )
)

# remove x-axis "Group" and configure text display
fig.update_layout(
    xaxis_title=None,
    yaxis_title="",
    width=400,
    height=400,
    showlegend=False,
    font=dict(size=16),  # Set font size to 14
)

fig.show()
# hack to fix mathjax

pio.kaleido.scope.mathjax = None
pdf_name = "articulation_professor.pdf"
# remove margins
fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
fig.write_image(pdf_name)
