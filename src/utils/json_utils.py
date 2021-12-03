import json

def read_json_lines(input_path):
    output = []
    with open(input_path, "r") as f:
        for line in f:
            output.append(json.loads(line))
    return output

def write_json_lines(list_to_write, output_path):
    with open(output_path, "w") as f:
        for line in list_to_write:
            json.dump(line, f)
            f.write("\n")