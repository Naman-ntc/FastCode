import sys
import json

input_jsons = sys.argv[1:-1]
output_json = sys.argv[-1]

combined_json = []
for input_json in input_jsons:
    with open(input_json, "r") as fp:
        input_json = json.load(fp)
        combined_json.extend(input_json)

with open(output_json, "w") as fp:
    json.dump(combined_json, indent=4, fp=fp)