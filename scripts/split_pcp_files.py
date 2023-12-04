import os
import csv
import argparse

def split_input(input_file, output_dir, batch_size):
    with open(input_file, 'r') as infile:
        reader = csv.reader(infile)
        header = next(reader)

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Initialize variables
        batch_num = 1
        current_batch = []

        for row in reader:
            current_batch.append(row)

            # Check if the batch size is reached
            if len(current_batch) == batch_size:
                output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(input_file))[0]}_{batch_num}.csv")
                with open(output_file, 'w', newline='') as outfile:
                    writer = csv.writer(outfile)
                    writer.writerow(header)
                    writer.writerows(current_batch)

                # Reset batch variables for the next batch
                current_batch = []
                batch_num += 1

        # Check for any remaining rows
        if current_batch:
            output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(input_file))[0]}_{batch_num}.csv")
            with open(output_file, 'w', newline='') as outfile:
                writer = csv.writer(outfile)
                writer.writerow(header)
                writer.writerows(current_batch)

def main():
    parser = argparse.ArgumentParser(description="Split input CSV file into batches.")
    parser.add_argument("input_file", help="Input CSV file to split.")
    parser.add_argument("output_dir", help="Output directory for batched CSV files.")
    parser.add_argument("batch_size", type=int, help="Batch size.")
    args = parser.parse_args()

    split_input(args.input_file, args.output_dir, args.batch_size)

if __name__ == "__main__":
    main()