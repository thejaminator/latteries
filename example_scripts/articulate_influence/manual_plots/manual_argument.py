import plotly.io as pio


import plotly.express as px
import pandas as pd

# Create dataframe
# mock data
alibaba_group = "Qwen"
google_group = "Gemini"
deepseek_group = "Deepseek"

"""
model	Argument
ITC: DeepSeek R1	34.1% (± 8.4%)
ITC: Gemini	47.4% (± 8.0%)
ITC: Qwen	15.8% (± 4.8%)
Claude-3.5-Sonnet	0.0% (± 0.0%)
Deepseek-Chat-v3	0.0% (± 0.0%)
GPT-4o	2.2% (± 2.1%)
Gemini-2.0-Flash-Exp	0.6% (± 1.1%)
Grok-2-1212	0.5% (± 1.0%)
Qwen-72b-Instruct	1.3% (± 1.5%)
"""
data = [
    {
        "Model": "GPT-4o",
        "Articulation Rate": 2.2,
        "Type": "Non-ITC",
        "Group": "GPT-4o",
        "Error": 2.1,
    },
    {
        "Model": "Qwen 72b",
        "Articulation Rate": 1.3,
        "Type": "Non-ITC",
        "Group": alibaba_group,
        "Error": 1.3,
    },
    {
        "Model": "QwQ",
        "Articulation Rate": 15.8,
        "Type": "Inference-Time-Compute (ITC)",
        "Group": alibaba_group,
        "Error": 4.8,
    },
    {
        "Model": "Gemini",
        "Articulation Rate": 0.6,
        "Type": "Non-ITC",
        "Group": google_group,
        "Error": 0.6,
    },
    {
        "Model": "Gemini-Thinking",
        "Articulation Rate": 47.4,
        "Type": "Inference-Time-Compute (ITC)",
        "Group": google_group,
        "Error": 8.0,
    },
    {
        "Model": "Deepseek-R1",
        "Articulation Rate": 34.1,
        "Type": "Inference-Time-Compute (ITC)",
        "Group": deepseek_group,
        "Error": 8.4,
    },
    {
        "Model": "Deepseek-V3",
        "Articulation Rate": 0.0,
        "Type": "Non-ITC",
        "Group": deepseek_group,
        "Error": 0.0,
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
fig.update_traces(textposition="outside", texttemplate=padd + "%{text:.0f}%", textfont_size=16, textangle=0)


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
    font=dict(size=12),  # Set font size to 14
    # tickfont=dict(size=16),
    titlefont=dict(size=16),
)

fig.show()
# hack to fix mathjax

pio.kaleido.scope.mathjax = None
pdf_name = "articulation_argument.pdf"
# remove margins
fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
fig.write_image(pdf_name)
