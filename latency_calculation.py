import re
from pathlib import Path
from typing import Dict, Optional

FLOAT = r'([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)'

PATTERNS = {
    "embedding_layer": rf'^\s*embedding_layer:\s*Total\s+{FLOAT}\s*ms',
    "average_decoder_layer": rf'^\s*Average Decoder Layer Runtime:\s*Total:\s*{FLOAT}\s*ms',
    "output_layer": rf'^\s*output_layer:\s*Total\s+{FLOAT}\s*ms',
}

def parse_runtimes(text: str) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "embedding_layer": None,
        "average_decoder_layer": None,
        "output_layer": None,
    }
    for key, pattern in PATTERNS.items():
        m = re.search(pattern, text, flags=re.MULTILINE)
        if m:
            out[key] = float(m.group(1))
    return out

# Example: read from file and assign to variables
def load_and_assign(path: str):
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    vals = parse_runtimes(text)

    # Assign to variables (falling back to 0.0 if missing, tweak as you like)
    embedding_layer = vals["embedding_layer"] or 0.0
    average_decoder_Layer = vals["average_decoder_layer"] or 0.0
    output_layer = vals["output_layer"] or 0.0

    return embedding_layer, average_decoder_Layer, output_layer

import sys
TP = int(sys.argv[1])
BS = int(sys.argv[2])

embed, middle, output = load_and_assign(f"/projects/bcrn/jshong/Megatron-LM/logs/a100_8/interm/all_nodes_LAYERS_{{32}}_tp_{{{TP}}}_bs_{{{BS}}}_STEP3_{{0}}.log")
# print(embed, middle, output)


middle_string = f"{middle}," * 32
print(embed + output + middle * 32)
print(f"[{embed},{middle_string}{output}]")