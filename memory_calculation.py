import re
from types import SimpleNamespace
from pathlib import Path

PEAK_START = r"---\s*PEAK MEMORY.*?---"
PEAK_END   = r"={5,}"  # the ===== line

# Matches: Pipeline Stage X (Rank Y): 123.45 MB/GB
LINE_RE = re.compile(
    r"^Pipeline\s+Stage\s+\d+\s+\(Rank\s+(\d+)\):\s*([\d,]+(?:\.\d+)?)\s*(MB|GB)\s*$",
    re.MULTILINE
)

def parse_peak_memory_mb(log_path, which_block = -1):
    text = Path(log_path).read_text(errors="ignore")

    # Find all PEAK MEMORY blocks
    blocks = re.findall(
        rf"{PEAK_START}(.*?){PEAK_END}",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )
    if not blocks:
        raise ValueError("No PEAK MEMORY block found.")

    block = blocks[which_block]  # default: last block

    ranks = {}
    for rank, val, unit in LINE_RE.findall(block):
        n = float(val.replace(",", ""))
        mb = n * (1024 if unit.upper() == "GB" else 1.0)
        ranks[f"Rank_{rank}"] = mb
    if not ranks:
        raise ValueError("PEAK MEMORY block found, but no per-rank lines parsed.")
    return ranks
import sys
TP = int(sys.argv[1])
BS = int(sys.argv[2])
MID_LAYERS = 8 // TP
# Example usage
A = parse_peak_memory_mb(f"/projects/bcrn/jshong/Megatron-LM/logs/a100_8/memory/all_nodes_LAYERS_{{0}}_tp_{{{TP}}}_bs_{{{BS}}}_STEP3_{{0}}.log")
B = parse_peak_memory_mb(f"/projects/bcrn/jshong/Megatron-LM/logs/a100_8/memory/all_nodes_LAYERS_{{{MID_LAYERS}}}_tp_{{{TP}}}_bs_{{{BS}}}_STEP3_{{0}}.log")
C = parse_peak_memory_mb(f"/projects/bcrn/jshong/Megatron-LM/logs/a100_8/memory/all_nodes_LAYERS_{{0}}_tp_{{{TP}}}_bs_{{{BS}}}_STEP3_{{1}}.log")
embed = int(sum(C.values()))
middle = (sum(B.values()) - sum(A.values())) / len(B.values()) * 32
middle_string = f"{(sum(B.values()) - sum(A.values())) / len(B.values())}," * 32
output = int((sum(A.values()) - sum(C.values())))
print(int(embed + output + middle))
print(f"[{embed},{middle_string}{output}]")

# A_Rank_0 = 10034
# A_Rank_1 = 450
# A_Rank_2 = 450 
# A_Rank_3 = 450 
# A_Rank_4 = 450
# A_Rank_5 = 450
# A_Rank_6 = 450 
# A_Rank_7 = 15050

# B_Rank_0 = 14878 
# B_Rank_1 = 6036 
# B_Rank_2 = 6036 
# B_Rank_3 = 6036 
# B_Rank_4 = 6036 
# B_Rank_5 = 6036 
# B_Rank_6 = 6036 
# B_Rank_7 = 21012 

# C_Rank_0 = 10034 
# C_Rank_1 = 450 
# C_Rank_2 = 450 
# C_Rank_3 = 450 
# C_Rank_4 = 450 
# C_Rank_5 = 450 
# C_Rank_6 = 450 
# C_Rank_7 = 450 

# embed = C_Rank_0 + C_Rank_1 + C_Rank_2 + C_Rank_3 + C_Rank_4 + C_Rank_5 + C_Rank_6 + C_Rank_7
# middle = (B_Rank_0 - A_Rank_0 + B_Rank_1 - A_Rank_1 + B_Rank_2 - A_Rank_2 + B_Rank_3 - A_Rank_3 + B_Rank_4 - A_Rank_4 + B_Rank_5 - A_Rank_5 + B_Rank_6 - A_Rank_6 + B_Rank_7 - A_Rank_7) / 8 * 32
# middle_string = f"{(B_Rank_0 - A_Rank_0 + B_Rank_1 - A_Rank_1 + B_Rank_2 - A_Rank_2 + B_Rank_3 - A_Rank_3 + B_Rank_4 - A_Rank_4 + B_Rank_5 - A_Rank_5 + B_Rank_6 - A_Rank_6 + B_Rank_7 - A_Rank_7) / 8}," * 32
# output = A_Rank_7 - C_Rank_7 + A_Rank_6 - C_Rank_6 + A_Rank_5 - C_Rank_5 + A_Rank_4 - C_Rank_4 + A_Rank_3 - C_Rank_3 + A_Rank_2 - C_Rank_2 + A_Rank_1 - C_Rank_1 + A_Rank_0 - C_Rank_0

# print(f"[{embed},{middle_string}{output}]")
# print(int(embed + output + middle))


# A_Rank_0 = 5022
# A_Rank_1 = 138
# A_Rank_2 = 138 
# A_Rank_3 = 6028 

# B_Rank_0 = 7350 
# B_Rank_1 = 2412 
# B_Rank_2 = 2412 
# B_Rank_3 = 8322 

# C_Rank_0 = 5022 
# C_Rank_1 = 114 
# C_Rank_2 = 114 
# C_Rank_3 = 114 

# embed = C_Rank_0 + C_Rank_1 + C_Rank_2 + C_Rank_3
# middle = (B_Rank_0 - A_Rank_0 + B_Rank_1 - A_Rank_1 + B_Rank_2 - A_Rank_2 + B_Rank_3 - A_Rank_3) / 4 * 32
# middle_string = f"{(B_Rank_0 - A_Rank_0 + B_Rank_1 - A_Rank_1 + B_Rank_2 - A_Rank_2 + B_Rank_3 - A_Rank_3) / 4}," * 32
# output = A_Rank_3 - C_Rank_3 + A_Rank_2 - C_Rank_2 + A_Rank_1 - C_Rank_1 + A_Rank_0 - C_Rank_0

# print(f"[{embed},{middle_string}{output}]")
# print(embed + output + middle)


# A_Rank_0 = 2514
# A_Rank_1 = 3018

# B_Rank_0 = 3674
# B_Rank_1 = 3966

# C_Rank_0 = 2514
# C_Rank_1 = 64
# embed = C_Rank_0 + C_Rank_1
# middle = (B_Rank_0 - A_Rank_0 + B_Rank_1 - A_Rank_1) / 2 * 32
# middle_string = f"{(B_Rank_0 - A_Rank_0 + B_Rank_1 - A_Rank_1) / 2}," * 32
# output = A_Rank_1 - C_Rank_1 + A_Rank_0 - C_Rank_0

# print(f"[{embed},{middle_string}{output}]")
# print(embed + output + middle)

# A_Rank_0 = 2650

# B_Rank_0 = 3114

# C_Rank_0 = 1262
# embed = C_Rank_0
# middle = (B_Rank_0 - A_Rank_0) * 32
# middle_string = f"{(B_Rank_0 - A_Rank_0)}," * 32
# output = A_Rank_0 - C_Rank_0

# print(f"[{embed},{middle_string}{output}]")
# print(embed + output + middle)