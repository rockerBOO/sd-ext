import json
import csv
import numpy


def to_json(results, file):
    """
    Save to file
    """
    with open(file, "w") as f:
        json.dump(results, f)


def from_json(file):
    """
    Load from file
    """
    with open(file) as f:
        results = json.load(f)

    return results


def to_numpy(results, file):
    """
    Save to numpy file
    """
    with open(file, "w") as f:
        numpy.save(results, f)


def from_numpy(file):
    """
    Load from numpy file
    """
    with open(file) as f:
        results = numpy.load(f)

    return results


def to_csv(results, file):
    """
    Save to csv file
    """
    with open(file, "w") as f:
        fieldnames = [key for key in results[0].keys()]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            writer.writerow(result)


def from_csv(file):
    """
    Load from csv file
    """
    with open(file) as f:
        reader = csv.DictReader(f)

        return {k: v for k, v in reader.items()}


def format_args(parser):
    parser.add_argument("--csv", help="Save to CSV file")
    parser.add_argument("--json", help="Save to JSON file")
    # parser.add_argument("--numpy", help="Save to numpy file")

    return parser
