import plotly.io as pio


import plotly.express as px
import pandas as pd

# Create dataframe
# mock data
alibaba_group = "Qwen"
google_group = "Gemini"


data = [
    # Professor
    {
        "Model": "Claude",
        "Articulation Rate": 6.7,
        "Type": "Best Non-ITC",
        "Group": "Professor",
        "Error": 5.2,
    },
    {
        "Model": "QwQ",
        "Articulation Rate": 46.9,
        "Type": "Qwen ITC",
        "Group": "Professor",
        "Error": 7.3,
    },
    {
        "Model": "Gemini-Thinking",
        "Articulation Rate": 52.4,
        "Type": "Gemini ITC",
        "Group": "Professor",
        "Error": 8.8,
    },
    ## Black Squares
    {
        "Model": "Gemini",
        "Articulation Rate": 28.1,
        "Type": "Gemini ITC",
        "Group": "Black<br>Squares",
        "Error": 6.3,
    },
    {
        "Model": "Qwen 72b",
        "Articulation Rate": 17.1,
        "Type": "Qwen ITC",
        "Group": "Black<br>Squares",
        "Error": 5.4,
    },
    {
        "Model": "Claude<br>Sonnet",
        "Articulation Rate": 3.1,
        "Type": "Best Non-ITC",
        "Group": "Black<br>Squares",
        "Error": 1.5,
    },
    ## White squares
    {
        "Model": "Gemini",
        "Articulation Rate": 25.3,
        "Type": "Gemini ITC",
        "Group": "White<br>Squares",
        "Error": 6.4,
    },
    {
        "Model": "Qwen 72b",
        "Articulation Rate": 14.2,
        "Type": "Qwen ITC",
        "Group": "White<br>Squares",
        "Error": 5.5,
    },
    {
        "Model": "Claude<br>Sonnet",
        "Articulation Rate": 3.1,
        "Type": "Best Non-ITC",
        "Group": "White<br>Squares",
        "Error": 2.1,
    },
    ## Argument
    {
        "Model": "Gemini",
        "Articulation Rate": 2.8,
        "Type": "Gemini ITC",
        "Group": "Argument",
        "Error": 2.8,
    },
    {
        "Model": "Qwen 72b",
        "Articulation Rate": 15.8,
        "Type": "Qwen ITC",
        "Group": "Argument",
        "Error": 4.8,
    },
    {
        "Model": "Claude",
        "Articulation Rate": 0.0,
        "Type": "Best Non-ITC",
        "Group": "Argument",
        "Error": 0.0,
    },
    ## Post-hoc
    {
        "Model": "Gemini",
        "Articulation Rate": 0.7,
        "Type": "Gemini ITC",
        "Group": "Post<br>Hoc",
        "Error": 0.9,
    },
    {
        "Model": "Qwen 72b",
        "Articulation Rate": 9.7,
        "Type": "Qwen ITC",
        "Group": "Post<br>Hoc",
        "Error": 3.8,
    },
    {
        "Model": "Claude",
        "Articulation Rate": 0.0,
        "Type": "Best Non-ITC",
        "Group": "Post<br>Hoc",
        "Error": 0.0,
    },
    ## Wrong Few-Shot
    {
        "Model": "Gemini",
        "Articulation Rate": 30.2,
        "Type": "Gemini ITC",
        "Group": "Wrong<br>Few-Shot",
        "Error": 5.6,
    },
    {
        "Model": "Qwen 72b",
        "Articulation Rate": 13.6,
        "Type": "Qwen ITC",
        "Group": "Wrong<br>Few-Shot",
        "Error": 5.3,
    },
    {
        "Model": "Claude",
        "Articulation Rate": 0.0,
        "Type": "Best Non-ITC",
        "Group": "Wrong<br>Few-Shot",
        "Error": 0.0,
    },
    ## Are you sure
    {
        "Model": "Gemini",
        "Articulation Rate": 3.2,
        "Type": "Gemini ITC",
        "Group": "Are you<br>sure?",
        "Error": 2.8,
    },
    {
        "Model": "Qwen 72b",
        "Articulation Rate": 2.2,
        "Type": "Qwen ITC",
        "Group": "Are you<br>sure?",
        "Error": 4.4,
    },
    {
        "Model": "Claude",
        "Articulation Rate": 0.0,
        "Type": "Best Non-ITC",
        "Group": "Are you<br>sure?",
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
    pattern_shape="Type",
    error_y="Error",
    barmode="group",
    hover_data=["Model"],
    text="Articulation Rate",
    color_discrete_sequence=["#636ef9", "#ef553b", "#ef553b"],  # Default blue, red, red
    pattern_shape_sequence=["", "", "/"],  # No pattern for first two, diagonal pattern for third
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
padd = "&nbsp;" * 10
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
pdf_name = "articulation_all.pdf"
# remove margins for PDF output only
fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
fig.write_image(pdf_name)
