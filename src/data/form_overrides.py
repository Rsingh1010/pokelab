"""Species whose default PokéAPI form is NOT the form used in battle.

Rule (set before looking at any model results):
    Override only if the default form is a temporary state that the Pokémon
    leaves automatically on entering battle, with no player action needed,
    and stays in the new form for most of the battle.

Each entry: species slug -> Pokémon slug to use instead of the default.
"""

FORM_OVERRIDES = {
    # Schooling ability: becomes School form on entry (above 25% HP).
    "wishiwashi": "wishiwashi-school",
    # Tera Shift ability: becomes Terastal form on entry. Normal form never fights.
    "terapagos": "terapagos-terastal",
}

# Considered and NOT overridden (documented so the choice is visible):
# - palafin: becomes Hero only after a switch-out (needs a player action).
# - zacian / zamazenta: Crowned form needs a held item (item choice).
# - aegislash, morpeko, meloetta: swap back and forth during battle.
# - minior, darmanitan-zen, zygarde-complete, eiscue: change late or at low HP.
