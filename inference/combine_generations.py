import ast
import sys
import json


def format_solution(solution):
    try:
        if (
            'if __name__ == "__main__":' in solution
            or "if __name__ == '__main__':" in solution
        ):
            try:
                astree = ast.parse(solution)
            except:
                return solution
            if not isinstance(astree.body[-1], ast.If):
                return solution

            ifblock = astree.body[-1]

            assert ast.unparse(ifblock.test) in [
                '__name__ == "__main__"',
                "__name__ == '__main__'",
            ], ast.unparse(ifblock.test)
            assert ifblock.orelse == [], ast.unparse(ifblock.orelse)

            astree.body = astree.body[:-1] + ifblock.body
            new_solution = ast.unparse(astree)
            return new_solution
        else:
            return solution
    except:
        return solution


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

    formatted_combined = [[format_solution(s) for s in sols] for sols in combined]
    with open(output_json.replace(".json", "_formatted.json"), "w") as fp:
        json.dump(formatted_combined, indent=4, fp=fp)

    return combined, formatted_combined


if __name__ == "__main__":
    input_jsons = sys.argv[1:-1]
    output_json = sys.argv[-1]
    main(input_jsons, output_json)
