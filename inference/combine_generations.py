import sys
import json


def main(input_jsons, output_json):
    combined_json = {}
    current_keys = set()
    for input_json in input_jsons:
        with open(input_json, "r") as fp:
            input_json = json.load(fp)
            input_json = {int(k): v for k, v in input_json.items()}
            keys = set(input_json.keys())
            if keys.intersection(current_keys):
                raise ValueError("Keys overlap")
            combined_json.update(input_json)

    ## sort on keys and remove keys
    combined = [combined_json[k] for k in sorted(combined_json)]

    with open(output_json, "w") as fp:
        json.dump(combined, indent=4, fp=fp)

    return combined


if __name__ == "__main__":
    input_jsons = sys.argv[1:-1]
    output_json = sys.argv[-1]
    main(input_jsons, output_json)
