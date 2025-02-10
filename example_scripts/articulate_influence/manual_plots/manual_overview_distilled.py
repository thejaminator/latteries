import plotly.io as pio


import plotly.express as px
import pandas as pd
from slist import Slist

from example_scripts.articulate_influence.run_articulation import RecallJsonlLine
from latteries.caller.openai_utils.shared import read_jsonl_file_into_basemodel

# open dump/recall.jsonl
# mock data

"""
e.g. {"model":"deepseek/deepseek-r1-distill-qwen-1.5b","bias_type":"Wrong Few-Shot","recall":0.02711864406779661,"error":0.018567196430797465}
{"model":"deepseek-reasoner","bias_type":"Professor","recall":0.59375,"error":0.06446175692130308}

"""

model_rename_map = {
    "deepseek-reasoner": "Deepseek R1<br>(Not Distilled)",
    "deepseek/deepseek-r1-distill-qwen-1.5b": "Qwen-1.5b",
    "deepseek/deepseek-r1-distill-qwen-14b": "Qwen-14b",
    "deepseek/deepseek-r1-distill-qwen-32b": "Qwen-32b",
    "deepseek/deepseek-r1-distill-llama-70b": "Llama-70b",
}

cue_rename_map = {
    "Wrong Few-Shot": "Wrong<br>Few-Shot",
    "Professor": "Professor",
    "Black Squares": "Black<br>Squares",
    "White Squares": "White<br>Squares",
    "Post-Hoc": "Post-<br>Hoc",
    "Argument": "Argument",
    "Are You Sure?": "Are You<br>Sure?",
}

model_sort_names = [
    "Qwen-1.5b",
    "Qwen-14b",
    "Qwen-32b",
    "Llama-70b",
    "Deepseek R1<br>(Not Distilled)",
]


_data: Slist[RecallJsonlLine] = read_jsonl_file_into_basemodel(
    path="dump/recall_distilled.jsonl", basemodel=RecallJsonlLine
).filter(
    # remove white squares
    lambda x: x.bias_type != "White Squares"
)
data = _data.map(lambda x: x.model_dump())

df = pd.DataFrame(data)
df["model"] = df["model"].map(model_rename_map)
df["bias_type"] = df["bias_type"].map(cue_rename_map)
# sort by model_sort_names
df["model"] = pd.Categorical(df["model"], categories=model_sort_names, ordered=True)
df = df.sort_values("model")

# Create bar plot with error bars
fig = px.bar(
    df,
    x="bias_type",
    y="recall",
    color="model",
    error_y="error",
    barmode="group",
    hover_data=["model"],
    text="recall",
    color_discrete_sequence=[
        "#636EFA",
        "#EF553B",
        "#00CC96",
        "#FFA15A",
        "#AB63FA",
        "#19D3F3",
        "#FF6692",
        "#B6E880",
        "#FF97FF",
        "#FECB52",
    ],
)

# Update font sizes for all text elements
fig.update_layout(
    font=dict(size=14),  # Base font size
    xaxis=dict(
        showline=True,
        linewidth=1,
        linecolor="black",
        tickfont=dict(size=14),
    ),
    yaxis=dict(
        showline=True,
        linewidth=1,
        linecolor="black",
        tickfont=dict(size=14),
        tickmode="array",
        tickvals=[0, 20, 40, 60],
        range=[0, 70],
    ),
    plot_bgcolor="white",
    paper_bgcolor="white",
)

# Update text labels with consistent padding and font size
padd = "&nbsp;" * 7
fig.update_traces(
    textposition="outside",
    texttemplate=padd + "%{text:.0f}%",
    textfont_size=12,
    textangle=0,
    textfont_family="Arial",
    error_y_thickness=1.0,  # Make error bars less thick
    error_y_width=2.0,  # Make error bars less wide
)

# Update secondary x-axis with consistent font size
fig.update_layout(
    xaxis2=dict(
        ticktext=df["model"],
        tickvals=list(range(len(df))),
        overlaying="x",
        side="bottom",
        title="Model",
        anchor="y",
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        tickfont=dict(size=14),
        titlefont=dict(size=14),
    )
)

# Final layout updates
fig.update_layout(
    xaxis_title=None,
    yaxis_title="",
    width=600,
    height=150,
    showlegend=False,
    margin=dict(l=0, r=0, t=0, b=50),
)

fig.show()
# hack to fix mathjax

pio.kaleido.scope.mathjax = None
pdf_name = "articulation_distilled.pdf"
# remove margins for PDF output only
fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
fig.write_image(pdf_name)
