import json
import os


def read_json_lines(input_path):
    output = []
    with open(input_path, "r") as f:
        for line in f:
            output.append(json.loads(line))
    return output


def write_json_lines(list_to_write, output_path, indent=None):
    with open(output_path, "w") as f:
        for line in list_to_write:
            json.dump(line, f, indent=indent)
            f.write("\n")


def append_json_lines(list_to_write, output_path, indent=None):
    with open(output_path, "a") as f:
        for line in list_to_write:
            json.dump(line, f, indent=indent)
            f.write("\n")


def write_to_json(X, y, output_folder_path, json_name, indent=None):
    list_to_write = [
        {"id": id, "label": int(label), "text": text}
        for id, (text, label) in enumerate(zip(X, y))
    ]
    write_json_lines(
        list_to_write=list_to_write,
        output_path=os.path.join(output_folder_path, json_name),
        indent=indent,
    )
