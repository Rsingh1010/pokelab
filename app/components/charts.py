"""Plotly figures shared by several pages."""
import numpy as np
import plotly.graph_objects as go

from app.theme import COLORS, TYPE_COLORS

AXES = {
    "role": ("speed_axis", "physical_axis", "← bulky · fast →", "← special · physical →"),
    "strength": ("strength_pc1", "strength_pc2", "← weaker · stronger →", "second strength component"),
}
COLOR_OPTIONS = {"BST": "bst", "Competitive usage": "usage_mean", "Anomaly score": "fe_relative_score",
                 "Primary type": "type_1", "Descriptive groups (k = 5)": "cluster_label"}


def stat_map(df, view="role", color_by="BST", highlight=None, explained=None, height=480):
    """Scatter of the role map (fully evolved, relative) or the strength map (all, standardized)."""
    x, y, xt, yt = AXES[view]
    d = df[df[x].notna()].copy()
    col = COLOR_OPTIONS[color_by]
    fig = go.Figure()
    hover = "<b>%{customdata[0]}</b><br>BST %{customdata[1]}<extra></extra>"
    custom = np.stack([d["name"], d["bst"]], axis=1)

    if col in ("type_1", "cluster_label"):
        palette = TYPE_COLORS if col == "type_1" else None
        for i, (g, part) in enumerate(d.groupby(col)):
            color = palette.get(g) if palette else ["#38BDF8", "#E63946", "#F59E0B", "#22C55E", "#A78BFA"][i % 5]
            fig.add_scatter(x=part[x], y=part[y], mode="markers", name=str(g),
                            marker=dict(size=6, color=color, opacity=.75),
                            customdata=np.stack([part["name"], part["bst"]], axis=1), hovertemplate=hover)
    else:
        values = d[col].fillna(0)
        if col == "usage_mean":
            values = np.log10(values + 1e-4)
        fig.add_scatter(x=d[x], y=d[y], mode="markers", name="",
                        marker=dict(size=6, color=values, colorscale=[[0, COLORS["border"]], [1, COLORS["blue"]]],
                                    opacity=.8, colorbar=dict(title=("log usage" if col == "usage_mean" else color_by),
                                                              thickness=10)),
                        customdata=custom, hovertemplate=hover, showlegend=False)

    if highlight is not None and highlight in set(d["name"]):
        h = d[d["name"] == highlight].iloc[0]
        fig.add_scatter(x=[h[x]], y=[h[y]], mode="markers", showlegend=False, hoverinfo="skip",
                        marker=dict(size=16, color=COLORS["red"], line=dict(width=2, color=COLORS["text"])))
        fig.add_annotation(x=h[x], y=h[y], text=f"<b>{h['name']}</b>", showarrow=True, arrowcolor=COLORS["text"],
                           ax=40, ay=-35, bgcolor=COLORS["card"], bordercolor=COLORS["border"], borderpad=4)
    sub = f" · first two components explain {explained:.0%} of the variation" if explained else ""
    fig.update_layout(height=height, xaxis_title=xt, yaxis_title=yt,
                      title=dict(text=f"{'Role map (fully evolved, stat shares)' if view == 'role' else 'Strength map (all Pokémon, raw stats)'}{sub}",
                                 font=dict(size=13)),
                      legend=dict(orientation="v", x=1.01, y=1))
    return fig
