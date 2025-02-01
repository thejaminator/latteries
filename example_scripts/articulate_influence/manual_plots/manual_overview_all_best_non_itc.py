import plotly.io as pio


import plotly.express as px
import pandas as pd

# Create dataframe
# mock data
alibaba_group = "Qwen"
google_group = "Gemini"
deepseek_group = "Deepseek"

"""
model	Professor	Black Squares	White Squares	Argument	Post-Hoc	Wrong Few-Shot	Are You Sure
ITC: DeepSeek R1	59.4% (± 6.4%)	25.2% (± 6.9%)	22.2% (± 8.2%)	34.1% (± 8.4%)	6.4% (± 4.6%)	25.4% (± 8.0%)	0.0% (± 0.0%)
ITC: Gemini	68.2% (± 9.8%)	35.0% (± 6.0%)	31.7% (± 7.2%)	47.4% (± 8.0%)	0.0% (± 0.0%)	48.8% (± 6.3%)	0.0% (± 0.0%)
ITC: Qwen	46.9% (± 7.3%)	17.1% (± 5.4%)	14.2% (± 5.5%)	15.8% (± 4.8%)	9.7% (± 3.8%)	13.6% (± 5.3%)	2.4% (± 4.7%)
Claude-3.5-Sonnet	6.7% (± 5.2%)	3.1% (± 1.5%)	3.1% (± 2.1%)	0.0% (± 0.0%)	0.0% (± 0.0%)	0.0% (± 0.0%)	0.0% (± 0.0%)
Deepseek-Chat-v3	6.5% (± 3.6%)	0.8% (± 1.1%)	2.3% (± 2.2%)	0.0% (± 0.0%)	0.0% (± 0.0%)	0.0% (± 0.0%)	0.0% (± 0.0%)
GPT-4o	2.4% (± 3.3%)	1.1% (± 1.5%)	0.0% (± 0.0%)	2.2% (± 2.1%)	0.0% (± 0.0%)	0.7% (± 1.3%)	0.0% (± 0.0%)
Gemini-2.0-Flash-Exp	13.0% (± 6.2%)	0.6% (± 1.1%)	0.0% (± 0.0%)	0.6% (± 1.1%)	0.0% (± 0.0%)	1.7% (± 1.6%)	0.0% (± 0.0%)
Grok-2-1212	4.9% (± 4.2%)	3.9% (± 2.2%)	0.4% (± 0.8%)	0.5% (± 1.0%)	0.0% (± 0.0%)	1.3% (± 1.3%)	0.0% (± 0.0%)
Llama-3.3-70b	7.7% (± 4.1%)	1.6% (± 1.8%)	0.0% (± 0.0%)	1.9% (± 1.8%)	0.0% (± 0.0%)	1.4% (± 1.4%)	0.0% (± 0.0%)
Qwen-72b-Instruct	5.3% (± 3.6%)	0.6% (± 1.2%)	0.6% (± 1.1%)	1.3% (± 1.5%)	0.0% (± 0.0%)	0.0% (± 0.0%)	0.0% (± 0.0%)
"""

data = [
    # Professor
    {
        "Model": "Gemini-2.0-Flash",
        "Articulation Rate": 13.0,
        "Type": "Best Non-ITC",
        "Group": "Professor",
        "Error": 6.2,
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
        "Articulation Rate": 68.2,
        "Type": "Gemini ITC",
        "Group": "Professor",
        "Error": 9.8,
    },
    {
        "Model": "Deepseek",
        "Articulation Rate": 59.4,
        "Type": "Deepseek ITC",
        "Group": "Professor",
        "Error": 6.4,
    },
    ## Black Squares
    {
        "Model": "Gemini",
        "Articulation Rate": 35.0,
        "Type": "Gemini ITC",
        "Group": "Black<br>Squares",
        "Error": 6.0,
    },
    {
        "Model": "Qwen 72b",
        "Articulation Rate": 17.1,
        "Type": "Qwen ITC",
        "Group": "Black<br>Squares",
        "Error": 5.4,
    },
    {
        "Model": "Deepseek",
        "Articulation Rate": 25.2,
        "Type": "Deepseek ITC",
        "Group": "Black<br>Squares",
        "Error": 6.9,
    },
    {
        "Model": "Grok-2",
        "Articulation Rate": 3.9,
        "Type": "Best Non-ITC",
        "Group": "Black<br>Squares",
        "Error": 2.2,
    },
    ## White squares
    {
        "Model": "Gemini",
        "Articulation Rate": 31.7,
        "Type": "Gemini ITC",
        "Group": "White<br>Squares",
        "Error": 7.2,
    },
    {
        "Model": "Qwen 72b",
        "Articulation Rate": 14.2,
        "Type": "Qwen ITC",
        "Group": "White<br>Squares",
        "Error": 5.5,
    },
    {
        "Model": "Deepseek",
        "Articulation Rate": 22.2,
        "Type": "Deepseek ITC",
        "Group": "White<br>Squares",
        "Error": 8.2,
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
        "Articulation Rate": 47.4,
        "Type": "Gemini ITC",
        "Group": "Argument",
        "Error": 8.0,
    },
    {
        "Model": "Qwen 72b",
        "Articulation Rate": 15.8,
        "Type": "Qwen ITC",
        "Group": "Argument",
        "Error": 4.8,
    },
    {
        "Model": "Deepseek",
        "Articulation Rate": 34.1,
        "Type": "Deepseek ITC",
        "Group": "Argument",
        "Error": 8.4,
    },
    {
        "Model": "Llama-3",
        "Articulation Rate": 1.9,
        "Type": "Best Non-ITC",
        "Group": "Argument",
        "Error": 1.8,
    },
    ## Post-hoc
    {
        "Model": "Gemini",
        "Articulation Rate": 0.0,
        "Type": "Gemini ITC",
        "Group": "Post<br>Hoc",
        "Error": 0.0,
    },
    {
        "Model": "Qwen 72b",
        "Articulation Rate": 9.7,
        "Type": "Qwen ITC",
        "Group": "Post<br>Hoc",
        "Error": 3.8,
    },
    {
        "Model": "Deepseek",
        "Articulation Rate": 6.4,
        "Type": "Deepseek ITC",
        "Group": "Post<br>Hoc",
        "Error": 4.6,
    },
    {
        "Model": "Claude<br>Sonnet",
        "Articulation Rate": 0.0,
        "Type": "Best Non-ITC",
        "Group": "Post<br>Hoc",
        "Error": 0.0,
    },
    ## Wrong Few-Shot
    {
        "Model": "Gemini",
        "Articulation Rate": 48.8,
        "Type": "Gemini ITC",
        "Group": "Wrong<br>Few-Shot",
        "Error": 6.3,
    },
    {
        "Model": "Qwen 72b",
        "Articulation Rate": 13.6,
        "Type": "Qwen ITC",
        "Group": "Wrong<br>Few-Shot",
        "Error": 5.3,
    },
    {
        "Model": "Deepseek",
        "Articulation Rate": 25.4,
        "Type": "Deepseek ITC",
        "Group": "Wrong<br>Few-Shot",
        "Error": 8.0,
    },
    {
        "Model": "Llama-3",
        "Articulation Rate": 1.4,
        "Type": "Best Non-ITC",
        "Group": "Wrong<br>Few-Shot",
        "Error": 1.4,
    },
    ## Are you sure
    {
        "Model": "Gemini",
        "Articulation Rate": 0.0,
        "Type": "Gemini ITC",
        "Group": "Are you<br>sure?",
        "Error": 0.0,
    },
    {
        "Model": "Qwen 72b",
        "Articulation Rate": 2.4,
        "Type": "Qwen ITC",
        "Group": "Are you<br>sure?",
        "Error": 4.7,
    },
    {
        "Model": "Deepseek",
        "Articulation Rate": 0.0,
        "Type": "Deepseek ITC",
        "Group": "Are you<br>sure?",
        "Error": 0.0,
    },
    {
        "Model": "Claude<br>Sonnet",
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
    # pattern_shape="Type",
    error_y="Error",
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
        tickvals=[0, 20, 40, 60, 80],
        range=[0, 85],
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
pdf_name = "articulation_all_non_itc.pdf"
# remove margins for PDF output only
fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
fig.write_image(pdf_name)
