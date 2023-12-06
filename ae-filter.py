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
    parser = argparse.ArgumentParser(
        description="Filter images using the aethestic score"
    )
    parser.add_argument(
        "scores_file", help="Scores CSV file to filter your results"
    )
    parser.add_argument(
        "--images_dir",
        required=True,
        help="Directory where the images are located for this CSV file for loading the images into the website",
    )
    args = parser.parse_args()
    main(args)
