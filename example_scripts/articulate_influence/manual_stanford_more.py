import plotly.io as pio


import plotly.express as px
import pandas as pd

# Create dataframe
# mock data
alibaba_group = "Qwen"
google_group = "Gemini"


data = [
    {
        "Model": "Claude-3.5-sonnet",
        "Articulation Rate": 6.7,
        "Type": "Non-ITC",
        "Group": "Claude<br>Sonnet",
        "Error": 5.2,
    },
    {
        "Model": "GPT-4o",
        "Articulation Rate": 2.4,
        "Type": "Non-ITC",
        "Group": "GPT-4o",
        "Error": 2.4,
    },
    {
        "Model": "grok-2",
        "Articulation Rate": 2.4,
        "Type": "Non-ITC",
        "Group": "Grok-2",
        "Error": 2.3,
    },
    {
        "Model": "Llama<br>70b",
        "Articulation Rate": 2.4,
        "Type": "Non-ITC",
        "Group": "Llama<br>70b",
        "Error": 2.4,
    },
    {
        "Model": "Qwen 72b",
        "Articulation Rate": 4.8,
        "Type": "Non-ITC",
        "Group": alibaba_group,
        "Error": 4.1,
    },
    {
        "Model": "QwQ",
        "Articulation Rate": 45.3,
        "Type": "Inference-Time-Compute (ITC)",
        "Group": alibaba_group,
        "Error": 9.0,
    },
    {
        "Model": "Gemini",
        "Articulation Rate": 14.8,
        "Type": "Non-ITC",
        "Group": google_group,
        "Error": 7.7,
    },
    {
        "Model": "Gemini-Thinking",
        "Articulation Rate": 50.6,
        "Type": "Inference-Time-Compute (ITC)",
        "Group": google_group,
        "Error": 11.2,
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
    hover_data=["Model"],
    text="Articulation Rate",
)

# Update font sizes for all text elements
fig.update_layout(
    font=dict(size=16),  # Base font size
    xaxis=dict(
        showline=True,
        linewidth=1,
        linecolor="black",
        tickfont=dict(size=16),
    ),
    yaxis=dict(
        showline=True,
        linewidth=1,
        linecolor="black",
        tickfont=dict(size=16),
        tickmode="array",
        tickvals=[0, 20, 40, 60],
        range=[0, 65],
    ),
    plot_bgcolor="white",
    paper_bgcolor="white",
)

# Update text labels with consistent padding and font size
padd = "&nbsp;" * 11
fig.update_traces(
    textposition="outside",
    texttemplate=padd + "%{text:.0f}%",
    textfont_size=12,
    textangle=0,
    textfont_family="Arial",
    # cliponaxis=False
)

# Update secondary x-axis with consistent font size
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
        tickfont=dict(size=16),
        titlefont=dict(size=16),
    )
)

# Final layout updates
fig.update_layout(
    xaxis_title=None, yaxis_title="", width=600, height=150, showlegend=False, margin=dict(l=0, r=0, t=50, b=50)
)

fig.show()
# hack to fix mathjax

pio.kaleido.scope.mathjax = None
pdf_name = "articulation_professor_more.pdf"
# remove margins for PDF output only
fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
fig.write_image(pdf_name)
