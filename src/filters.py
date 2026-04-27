def apply_filters(
    df,
    include_legendary=True,
    include_mythical=True,
    include_ultrabeast=True,
    include_ordinary=True,
    generations=None
):

    filtered = df.copy()

    # ______________________________________________________________________________________
    # Categorie Pokémon
    # ______________________________________________________________________________________
    if not include_legendary:
        filtered = filtered[filtered["is_legendary"] == 0]

    if not include_mythical:
        filtered = filtered[filtered["is_mythical"] == 0]

    if not include_ultrabeast:
        filtered = filtered[filtered["is_ultrabeast"] == 0]

    if not include_ordinary:
        filtered = filtered[filtered["is_ordinary"] == 0]

    # ______________________________________________________________________________________
    # Generazioni
    # ______________________________________________________________________________________
    if generations:
        filtered = filtered[filtered["generation"].isin(generations)]

    return filtered

def get_pokemon_category(row):
    # Restituisce una categoria leggibile del Pokémon.

    if row.get("is_legendary", 0) == 1:
        return "Leggendario"

    if row.get("is_mythical", 0) == 1:
        return "Mitico"

    if row.get("is_ultrabeast", 0) == 1:
        return "Ultra Creatura"

    return "Ordinario"