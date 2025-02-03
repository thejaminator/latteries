import plotly.io as pio


import plotly.express as px
import pandas as pd

# Create dataframe
# mock data
alibaba_group = "Qwen"
google_group = "Gemini"
deepseek_group = "Deepseek"

"""

"""
data = [
    # Professor
    {
        "Model": "Gemini-2.0-Flash",
        "Articulation Rate": 22.4,
        "Type": "Best Non-ITC",
        "Group": "Professor",
        "Error": 0,
    },
    {
        "Model": "QwQ",
        "Articulation Rate": 55.4,
        "Type": "Qwen ITC",
        "Group": "Professor",
        "Error": 0,
    },
    {
        "Model": "Gemini-Thinking",
        "Articulation Rate": 35.8,
        "Type": "Gemini ITC",
        "Group": "Professor",
        "Error": 0,
    },
    {
        "Model": "Deepseek",
        "Articulation Rate": 61.1,
        "Type": "Deepseek ITC",
        "Group": "Professor",
        "Error": 0,
    },
    ## Black Squares
    {
        "Model": "Gemini",
        "Articulation Rate": 50.1,
        "Type": "Gemini ITC",
        "Group": "Black<br>Squares",
        "Error": 0,
    },
    {
        "Model": "Qwen 72b",
        "Articulation Rate": 28.6,
        "Type": "Qwen ITC",
        "Group": "Black<br>Squares",
        "Error": 0,
    },
    {
        "Model": "Deepseek",
        "Articulation Rate": 36.8,
        "Type": "Deepseek ITC",
        "Group": "Black<br>Squares",
        "Error": 0,
    },
    {
        "Model": "Grok-2",
        "Articulation Rate": 7.5,
        "Type": "Best Non-ITC",
        "Group": "Black<br>Squares",
        "Error": 0,
    },
    ## White squares
    {
        "Model": "Gemini",
        "Articulation Rate": 46.6,
        "Type": "Gemini ITC",
        "Group": "White<br>Squares",
        "Error": 0,
    },
    {
        "Model": "Qwen 72b",
        "Articulation Rate": 24.3,
        "Type": "Qwen ITC",
        "Group": "White<br>Squares",
        "Error": 0,
    },
    {
        "Model": "Deepseek",
        "Articulation Rate": 35.2,
        "Type": "Deepseek ITC",
        "Group": "White<br>Squares",
        "Error": 0,
    },
    {
        "Model": "Claude<br>Sonnet",
        "Articulation Rate": 5.9,
        "Type": "Best Non-ITC",
        "Group": "White<br>Squares",
        "Error": 0,
    },
    ## Argument
    {
        "Model": "Gemini",
        "Articulation Rate": 46.9,
        "Type": "Gemini ITC",
        "Group": "Argument",
        "Error": 0,
    },
    {
        "Model": "Qwen 72b",
        "Articulation Rate": 25.9,
        "Type": "Qwen ITC",
        "Group": "Argument",
        "Error": 0,
    },
    {
        "Model": "Deepseek",
        "Articulation Rate": 36.5,
        "Type": "Deepseek ITC",
        "Group": "Argument",
        "Error": 0,
    },
    {
        "Model": "GPT-4o",
        "Articulation Rate": 4.2,
        "Type": "Best Non-ITC",
        "Group": "Argument",
        "Error": 0,
    },
    ## Post-hoc
    {
        "Model": "Gemini",
        "Articulation Rate": 0.0,
        "Type": "Gemini ITC",
        "Group": "Post<br>Hoc",
        "Error": 0,
    },
    {
        "Model": "Qwen 72b",
        "Articulation Rate": 17.1,
        "Type": "Qwen ITC",
        "Group": "Post<br>Hoc",
        "Error": 0,
    },
    {
        "Model": "Deepseek",
        "Articulation Rate": 8.1,
        "Type": "Deepseek ITC",
        "Group": "Post<br>Hoc",
        "Error": 0,
    },
    {
        "Model": "Claude<br>Sonnet",
        "Articulation Rate": 0.0,
        "Type": "Best Non-ITC",
        "Group": "Post<br>Hoc",
        "Error": 0,
    },
    ## Wrong Few-Shot
    {
        "Model": "Gemini",
        "Articulation Rate": 61.7,
        "Type": "Gemini ITC",
        "Group": "Wrong<br>Few-Shot",
        "Error": 0,
    },
    {
        "Model": "Qwen 72b",
        "Articulation Rate": 21.3,
        "Type": "Qwen ITC",
        "Group": "Wrong<br>Few-Shot",
        "Error": 0,
    },
    {
        "Model": "Deepseek",
        "Articulation Rate": 30.1,
        "Type": "Deepseek ITC",
        "Group": "Wrong<br>Few-Shot",
        "Error": 0,
    },
    {
        "Model": "Gemini-2.0-Flash",
        "Articulation Rate": 3.3,
        "Type": "Best Non-ITC",
        "Group": "Wrong<br>Few-Shot",
        "Error": 0,
    },
    ## Are you sure
    {
        "Model": "Gemini",
        "Articulation Rate": 0.0,
        "Type": "Gemini ITC",
        "Group": "Are you<br>sure?",
        "Error": 0,
    },
    {
        "Model": "Qwen 72b",
        "Articulation Rate": 0.0,
        "Type": "Qwen ITC",
        "Group": "Are you<br>sure?",
        "Error": 0,
    },
    {
        "Model": "Deepseek",
        "Articulation Rate": 0.0,
        "Type": "Deepseek ITC",
        "Group": "Are you<br>sure?",
        "Error": 0,
    },
    {
        "Model": "Claude<br>Sonnet",
        "Articulation Rate": 0.0,
        "Type": "Best Non-ITC",
        "Group": "Are you<br>sure?",
        "Error": 0,
    },
]
df = pd.DataFrame(data)

# Create bar plot with error bars
fig = px.bar(
    df,
    x="Group",
    y="Articulation Rate",
    color="Type",
    # pattern_shape="Type",
    # error_y="Error",
    barmode="group",
    hover_data=["Model"],
    text="Articulation Rate",
    # color_discrete_sequence=["#636ef9", "#ef553b", "#ef553b"],  # Default blue, red, red
    # pattern_shape_sequence=["", "", "/"],  # No pattern for first two, diagonal pattern for third
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
        range=[0, 80],
    ),
    plot_bgcolor="white",
    paper_bgcolor="white",
)

# Update text labels with consistent padding and font size
padd = ""
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
    xaxis_title=None, yaxis_title="", width=600, height=200, showlegend=False, margin=dict(l=0, r=0, t=0, b=50)
)

fig.show()
# hack to fix mathjax

pio.kaleido.scope.mathjax = None
pdf_name = "articulation_f1.pdf"
# remove margins for PDF output only
fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
fig.write_image(pdf_name)
