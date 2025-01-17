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
        "Model": "Claude-3.5-Sonnet",
        "Articulation Rate": 8.9,
        "Type": "Best Non-ITC",
        "Group": "Professor",
    },
    {
        "Model": "Qwen",
        "Articulation Rate": 55.4,
        "Type": "Qwen ITC",
        "Group": "Professor",
    },
    {
        "Model": "Gemini",
        "Articulation Rate": 44.7,
        "Type": "Gemini ITC",
        "Group": "Professor",
    },
    ## Black Squares
    {
        "Model": "Grok-2-1212",
        "Articulation Rate": 7.5,
        "Type": "Best Non-ITC",
        "Group": "Black<br>Squares",
    },
    {
        "Model": "Qwen",
        "Articulation Rate": 28.6,
        "Type": "Qwen ITC",
        "Group": "Black<br>Squares",
    },
    {
        "Model": "Gemini",
        "Articulation Rate": 42.4,
        "Type": "Gemini ITC",
        "Group": "Black<br>Squares",
    },
    ## White squares
    {
        "Model": "Claude-3.5-Sonnet",
        "Articulation Rate": 5.9,
        "Type": "Best Non-ITC",
        "Group": "White<br>Squares",
    },
    {
        "Model": "Qwen",
        "Articulation Rate": 24.3,
        "Type": "Qwen ITC",
        "Group": "White<br>Squares",
    },
    {
        "Model": "Gemini",
        "Articulation Rate": 39.1,
        "Type": "Gemini ITC",
        "Group": "White<br>Squares",
    },
    ## Argument
    {
        "Model": "GPT-4o",
        "Articulation Rate": 4.2,
        "Type": "Best Non-ITC",
        "Group": "Argument",
    },
    {
        "Model": "Qwen",
        "Articulation Rate": 25.9,
        "Type": "Qwen ITC",
        "Group": "Argument",
    },
    {
        "Model": "Gemini",
        "Articulation Rate": 5.5,
        "Type": "Gemini ITC",
        "Group": "Argument",
    },
    ## Post-hoc
    {
        "Model": "Claude-3.5-Sonnet",
        "Articulation Rate": 0.0,
        "Type": "Best Non-ITC",
        "Group": "Post<br>Hoc",
    },
    {
        "Model": "Qwen",
        "Articulation Rate": 17.1,
        "Type": "Qwen ITC",
        "Group": "Post<br>Hoc",
    },
    {
        "Model": "Gemini",
        "Articulation Rate": 1.3,
        "Type": "Gemini ITC",
        "Group": "Post<br>Hoc",
    },
    ## Wrong Few-Shot
    {
        "Model": "Grok-2-1212",
        "Articulation Rate": 2.6,
        "Type": "Best Non-ITC",
        "Group": "Wrong<br>Few-Shot",
    },
    {
        "Model": "Qwen",
        "Articulation Rate": 21.3,
        "Type": "Qwen ITC",
        "Group": "Wrong<br>Few-Shot",
    },
    {
        "Model": "Gemini",
        "Articulation Rate": 44.3,
        "Type": "Gemini ITC",
        "Group": "Wrong<br>Few-Shot",
    },
    ## Are you sure
    {
        "Model": "Claude-3.5-Sonnet",
        "Articulation Rate": 0.0,
        "Type": "Best Non-ITC",
        "Group": "Are you<br>sure?",
    },
    {
        "Model": "Qwen",
        "Articulation Rate": 0.0,
        "Type": "Qwen ITC",
        "Group": "Are you<br>sure?",
    },
    {
        "Model": "Gemini",
        "Articulation Rate": 5.6,
        "Type": "Gemini ITC",
        "Group": "Are you<br>sure?",
    },
]
df = pd.DataFrame(data)

# Create bar plot without error bars
fig = px.bar(
    df,
    x="Group",
    y="Articulation Rate",
    color="Type",
    pattern_shape="Type",
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
# padd = "&nbsp;" * 10
fig.update_traces(
    textposition="outside",
    texttemplate="%{text:.1f}%",
    textfont_size=12,
    textangle=0,
    textfont_family="Arial",
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
pdf_name = "f1_all.pdf"
# remove margins for PDF output only
fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
fig.write_image(pdf_name)
