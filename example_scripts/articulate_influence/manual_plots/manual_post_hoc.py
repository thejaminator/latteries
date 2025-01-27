from typing import Sequence
import plotly.io as pio


import plotly.express as px
import pandas as pd

# Create dataframe
# mock data
alibaba_group = "Qwen"
google_group = "Gemini"
deepseek_group = "Deepseek"

"""
model	Post-Hoc
ITC: DeepSeek R1	6.4% (± 4.6%)
ITC: Gemini	0.0% (± 0.0%)
ITC: Qwen	9.7% (± 3.8%)
Claude-3.5-Sonnet	0.0% (± 0.0%)
Deepseek-Chat-v3	0.0% (± 0.0%)
GPT-4o	0.0% (± 0.0%)
Gemini-2.0-Flash-Exp	0.0% (± 0.0%)
Grok-2-1212	0.0% (± 0.0%)
Llama-3.3-70b	0.0% (± 0.0%)
Qwen-72b-Instruct	0.0% (± 0.0%)
"""
MANUAL_DATA = [
    {
        "Model": "Qwen 72b",
        "Articulation Rate": 0.0,
        "Type": "Non-ITC",
        "Group": alibaba_group,
        "Error": 0.0,
    },
    {
        "Model": "QwQ",
        "Articulation Rate": 9.7,
        "Type": "Inference-Time-Compute (ITC)",
        "Group": alibaba_group,
        "Error": 3.8,
    },
    {
        "Model": "Gemini",
        "Articulation Rate": 0.0,
        "Type": "Non-ITC",
        "Group": google_group,
        "Error": 0.0,
    },
    {
        "Model": "Gemini-Thinking",
        "Articulation Rate": 0.0,
        "Type": "Inference-Time-Compute (ITC)",
        "Group": google_group,
        "Error": 0.0,
    },
    {
        "Model": "Deepseek-R1",
        "Articulation Rate": 6.4,
        "Type": "Inference-Time-Compute (ITC)",
        "Group": deepseek_group,
        "Error": 4.6,
    },
    {
        "Model": "Deepseek-V3",
        "Articulation Rate": 0.0,
        "Type": "Non-ITC",
        "Group": deepseek_group,
        "Error": 0.0,
    },
]


def plot_manual_stanford_fig_1(data: Sequence[dict] = []):
    if not data:
        data = MANUAL_DATA
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
        # check y ticks to 0, 10, 20
        yaxis=dict(
            tickmode="array",
            tickvals=[0, 10, 15],
            range=[0, 15],
        ),
    )

    fig.show()
    # hack to fix mathjax

    pio.kaleido.scope.mathjax = None
    pdf_name = "articulation_post_hoc.pdf"
    # remove margins
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    fig.write_image(pdf_name)


if __name__ == "__main__":
    plot_manual_stanford_fig_1()
