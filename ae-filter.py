import csv
import argparse


def get_info(file):
    results = []

    with open(file) as f:
        reader = csv.reader(f)

        for i, row in enumerate(reader):
            # skip header row
            if i == 0:
                continue

            results.append({"file": row[0], "score": float(row[1])})

    return results


def by_score(min, max):
    return lambda result: result["score"] >= min and result["score"] <= max


def filter_by_score_range(results, min, max):
    return [x for x in filter(by_score(min, max), results)]
    # return [item for item in results if by_score(min, max)(item)]


def main(args):
    results = get_info(args.scores_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "scores_file", help="Scores CSV file to filter your results"
    )
    parser.add_argument(
        "--images_dir",
        required=True,
        help="Directory where the images are located for this CSV file for loading the images into the website",
    )
    parser.add_argument(
        "--server",
        default=False,
        action="store_true",
        help="Run a webserver to view filtering in your browser.",
    )
    parser.add_argument(
        "--port",
        default=3456,
        type=int,
        help="Set the port to run the server on.",
    )
    args = parser.parse_args()
    main(args)
