"""Small reusable pieces of HTML. Keeps pages short and the look consistent."""
import html

import streamlit as st

from app.theme import COLORS, TYPE_COLORS

STAT_LABELS = {"hp": "HP", "attack": "Attack", "defense": "Defense",
               "sp_attack": "Sp. Atk", "sp_defense": "Sp. Def", "speed": "Speed"}


def type_badges(type_1, type_2=None) -> str:
    out = ""
    for t in (type_1, type_2):
        if isinstance(t, str):
            out += f'<span class="pl-badge" style="background:{TYPE_COLORS.get(t, "#999")}">{t}</span>'
    return out


def stat_bars(row, max_value=255) -> str:
    rows = ""
    for col, label in STAT_LABELS.items():
        v = int(row[col])
        color = COLORS["red"] if v < 60 else COLORS["amber"] if v < 90 else COLORS["green"] if v < 120 else COLORS["blue"]
        rows += (f'<div class="pl-stat-row"><span class="pl-stat-name">{label}</span>'
                 f'<span class="pl-stat-val">{v}</span><div class="pl-stat-bar">'
                 f'<div class="pl-stat-fill" style="width:{v / max_value * 100:.0f}%;background:{color}"></div>'
                 f'</div></div>')
    return rows


def card(title: str, body_html: str):
    st.markdown(f'<div class="pl-card"><h4>{html.escape(title)}</h4>{body_html}</div>', unsafe_allow_html=True)


def kpi(value, label: str):
    st.markdown(f'<div class="pl-card"><div class="pl-kpi">{value}</div>'
                f'<div class="pl-kpi-label">{html.escape(label)}</div></div>', unsafe_allow_html=True)


def note(text: str):
    st.markdown(f'<div class="pl-note">{text}</div>', unsafe_allow_html=True)


def status(s: str) -> str:
    cls = "pl-status-done" if s.startswith("completed") else "pl-status-pending"
    return f'<span class="{cls}">{html.escape(s.upper())}</span>'


def dex_no(n) -> str:
    return f'<span class="pl-mono">#{int(n):04d}</span>'


def wordmark():
    st.markdown(f'''<div style="display:flex;align-items:center;gap:.6rem;margin:.2rem 0 .8rem">
      <div style="width:14px;height:14px;border-radius:50%;background:{COLORS["red"]};
                  box-shadow:0 0 0 4px rgba(230,57,70,.18)"></div>
      <div><div style="font-weight:700;font-size:1.35rem;line-height:1">PokéLab</div>
      <div class="pl-muted" style="font-size:.75rem">Explore · Analyze · Optimize</div></div></div>''',
                unsafe_allow_html=True)


def _is_true(v) -> bool:
    return v is True or str(v) == "True"


def describe_group(label) -> str:
    """'speed+ / defense-' -> 'Higher Speed, lower Defense'."""
    if not isinstance(label, str):
        return ""
    up, down = [s.strip().rstrip("+-") for s in label.split("/")]
    return f"Higher {STAT_LABELS.get(up, up)}, lower {STAT_LABELS.get(down, down)}"


def profile_card(row, n_fully_evolved: int):
    """The Pokédex-style card. Everything shown comes from saved results."""
    left, right = st.columns([2, 3])
    with left:
        if isinstance(row.get("artwork_url"), str):
            st.image(row["artwork_url"], width=200)
        roles = [r for r in ("fast", "bulky", "physical", "special") if _is_true(row.get(r))]
        tags = "".join(f'<span class="pl-tag">{r}</span>' for r in roles) or \
            '<span class="pl-muted">no strong role tag</span>'
        if row.get("is_fully_evolved"):
            group = (f'<div class="pl-muted" style="margin-top:.5rem">Descriptive group:<br>'
                     f'{html.escape(describe_group(row.get("cluster_label")))}</div>')
            body = f'<div class="pl-kpi-label">Role tags</div>{tags}{group}'
        else:
            body = '<span class="pl-muted">Not fully evolved: outside the role analysis.</span>'
        st.markdown(f'<div class="pl-card">{body}</div>', unsafe_allow_html=True)
    with right:
        legend = " · legendary" if row.get("is_legendary") else " · mythical" if row.get("is_mythical") else ""
        st.markdown(
            f'<h2 style="margin:0">{html.escape(row["name"])}</h2>'
            f'{dex_no(row["dex_id"])} <span class="pl-muted">· Gen {int(row["generation"])}{legend}</span>'
            f'<div style="margin:.5rem 0 .8rem">{type_badges(row["type_1"], row.get("type_2"))}</div>'
            f'{stat_bars(row)}'
            f'<div class="pl-stat-row" style="margin-top:.4rem"><span class="pl-stat-name">Total</span>'
            f'<span class="pl-stat-val">{int(row["bst"])}</span></div>',
            unsafe_allow_html=True)
        usage = row.get("usage_mean")
        if not row.get("is_fully_evolved"):
            use_txt = '<span class="pl-muted">not tracked (not fully evolved)</span>'
        elif not _is_true(row.get("legal")):
            use_txt = f'<span class="pl-muted">banned in Nat Dex ({row.get("natdex_tier")})</span>'
        else:
            use_txt = f'<b>{usage:.2%}</b> <span class="pl-muted">of teams · tier {row.get("natdex_tier")}</span>'
        if row.get("is_fully_evolved") and row.get("fe_relative_rank") == row.get("fe_relative_rank"):
            anom = f'#{int(row["fe_relative_rank"])} of {n_fully_evolved} <span class="pl-muted">(stat shape)</span>'
        else:
            anom = f'#{int(row["raw_rank"])} of 1,025 <span class="pl-muted">(raw stats)</span>'
        st.markdown(
            f'<div class="pl-card"><div class="pl-kpi-label">Competitive usage · Gen 9 Nat Dex</div>{use_txt}'
            f'<div class="pl-kpi-label" style="margin-top:.6rem">Anomaly rank</div>{anom}</div>',
            unsafe_allow_html=True)


def select(name: str):
    """Change the selected Pokémon from a button or table click.

    Must be used as an on_click / on_select CALLBACK: Streamlit only allows changing a
    widget's value (the sidebar search) before the page re-runs, which is when callbacks fire.
    """
    st.session_state["selected_name"] = name


def matchup_badges(profile) -> str:
    """Defensive matchups grouped by multiplier, as type badges."""
    groups = [(4, "4× weak"), (2, "2× weak"), (0.5, "½× resists"), (0.25, "¼× resists"), (0, "immune")]
    out = ""
    for mult, label in groups:
        types = [t for t, m in profile.items() if m == mult]
        if types:
            out += (f'<div style="margin:.35rem 0"><span class="pl-kpi-label" style="display:inline-block;width:90px">'
                    f'{label}</span>{"".join(type_badges(t) for t in types)}</div>')
    return out
