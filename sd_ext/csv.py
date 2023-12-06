import csv


def save_to_csv(results: dict, outfile):
    with open(outfile, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=results.keys())
        writer.writeheader()
        for result in results:
            writer.writerow(result)
