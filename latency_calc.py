import json
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Read execution time from profile JSON.')
parser.add_argument('--tp', type=int, required=True, help='Tensor Parallel size')
parser.add_argument('--bs', type=int, required=True, help='Batch size')
args = parser.parse_args()

# File path and JSON loading
file_path = f'/projects/bcrn/jshong/profile_LLama/DeviceType.A40_tp{args.tp}_bs{args.bs}.json'
with open(file_path, 'r') as file:
    data = json.load(file)
    total_time = sum(data['execution_time']['layer_compute_total_ms'])
    print(f"Total Execution Time (ms): {total_time}")