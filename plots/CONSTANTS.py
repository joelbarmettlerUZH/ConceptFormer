import math


COLOR_BY_MODEL_BASE = {
    "gpt2": "#264653",
    "llama2": "#f4a261"
}

COLOR_BY_DATASET = {
    "trex": "#147960",
    "trex-lite": "#10a37f"
}

COLOR_BY_MODEL_NAME = {
    "TheBloke/Llama-2-7B-GPTQ": "#ee8959",
    "openlm-research/open_llama_3b_v2": COLOR_BY_MODEL_BASE["llama2"],
    "gpt2-xl": "#e9c46a",
    "gpt2-large": "#2a9d8f",
    "gpt2-medium": "#287271",
    "gpt2": COLOR_BY_MODEL_BASE['gpt2'],
}

COLOR_BY_CONCEPTFORMER = {
    1: "#81cdc6",
    2: "#4fb9af",
    3: "#28a99e",
    4: "#05998c",
    5: "#048c7f",
    10: "#037c6e",
    15: "#036c5f",
    20: "#025043",
}

ALIAS_BY_MODEL_NAME = {
    "TheBloke/Llama-2-7B-GPTQ": "LLaMA-2 7B",
    "openlm-research/open_llama_3b_v2": "LLaMA-2 3B",
    "gpt2-xl": "GPT-2 1.5B",
    "gpt2-large": "GPT-2 0.7B",
    "gpt2-medium": "GPT-2 0.3B",
    "gpt2": "GPT-2 0.1B",
}

FORMAT_ALIAS_BY_DESC = {
    "Subject, {predicate_1} {object_1}, {predicate_2} {object_2}, ...": "Inline",
    "Subject ({predicate_1}: {object_1}, {predicate_2}: {object_2}, ...)": "Brackets",
}

LINESTYLE ={
    "base": "-",
    "textinjection": "--",
    "pseudowords": "-.",
}

DATASET_ALIAS = {
    'TriREx': 'Tri-REx',
    'TRExBite': 'T-Rex Bite',
    'TriRexLite': 'Tri-REx Lite',
    'TRExBiteLite': 'T-Rex Bite Lite'
}

def greyscale_hex(value):
    """
    Generates a greyscale hex color value based on an input value between 1 and 100.
    The scale is logarithmic, making the darkest value lighter and the lightest value darker.
    """
    if value < 1 or value > 100:
        raise ValueError("Value must be between 1 and 100")

    # Using a logarithmic scale for color mapping
    # The constants are adjusted to fit the logarithmic scale within the desired range
    min_log = math.log(1)
    max_log = math.log(100)
    log_range = max_log - min_log
    normalized_value = (math.log(value) - min_log) / log_range

    # Adjust the greyscale range (e.g., 100 for darkest, 205 for lightest)
    greyscale_value = int(0 + normalized_value * 205)

    # Convert to hexadecimal
    hex_value = f'#{greyscale_value:02x}{greyscale_value:02x}{greyscale_value:02x}'

    return hex_value

def hex_to_rgba(hex_color, value):
    """
    Converts a hex color value to an RGBA color value with varying levels of opacity.
    The opacity is determined by a logarithmic scale based on an input value between 1 and 100.
    """
    if value < 1 or value > 100:
        raise ValueError("Value must be between 1 and 100")

    if len(hex_color) != 7 or not hex_color.startswith('#'):
        raise ValueError("Invalid hex color format")

    # Extract RGB components from the hex color
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)

    # Using a logarithmic scale for opacity mapping
    min_log = math.log(1)
    max_log = math.log(100)
    log_range = max_log - min_log
    normalized_value = (math.log(value) - min_log) / log_range

    # Calculate opacity
    # Opacity interpolates from 1 to 0.1 as value goes from 1 to 100
    opacity = 1 - normalized_value * 0.9

    return (r / 256, g / 256, b / 256, opacity)