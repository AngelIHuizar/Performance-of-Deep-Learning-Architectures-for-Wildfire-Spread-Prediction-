"""
shared/variable_groups.py

Agrupacion de INPUT_FEATURES en los 5 grupos que pide el protocolo de
ablacion del plan de extension (Semana 1-2, "Protocolo de ablacion de
variables"): (1) historial previo del incendio, (2) meteorologia,
(3) vegetacion/combustible, (4) terreno/topografia, (5) proxy poblacional.
"""

VARIABLE_GROUPS = {
    "historial_incendio": ["viirs_PrevFireMask"],
    "meteorologia": ["wind_75", "wind_avg", "tmp_75", "tmp_day", "avg_sph", "pr", "erc", "pdsi"],
    "vegetacion": ["NDVI"],
    "topografia": ["elevation"],
    "poblacion": ["population"],
}

# Verificacion de que los grupos cubren exactamente INPUT_FEATURES, sin
# huecos ni duplicados -- se corre al importar este modulo.
def _validar_grupos(input_features):
    todas = [f for grupo in VARIABLE_GROUPS.values() for f in grupo]
    faltantes = set(input_features) - set(todas)
    sobrantes = set(todas) - set(input_features)
    duplicadas = [f for f in todas if todas.count(f) > 1]
    if faltantes:
        raise ValueError(f"Variables sin grupo asignado: {faltantes}")
    if sobrantes:
        raise ValueError(f"Grupos mencionan variables que no existen: {sobrantes}")
    if duplicadas:
        raise ValueError(f"Variables asignadas a mas de un grupo: {set(duplicadas)}")
