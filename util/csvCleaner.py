import csv
import numpy as np

class CSVCleaner:
    def __init__(self, filename: str):
        """Loads and cleans a CSV file"""
        self.filename = filename
        self.data = []
        self.header = []
        self.encoders = {}

    def load_csv(self) -> None:
        """Loads CSV data and converts headers to lowercase"""
        with open(self.filename, "r", newline="") as file:
            reader = csv.reader(file)
            self.header = next(reader)
            # Normalize headers to lowercase
            self.header = [col.strip().lower() for col in self.header]
            for row in reader:
                # Skip empty rows
                if any(cell.strip() for cell in row):
                    self.data.append(row)

    def encode_data(self) -> np.ndarray:
        """Converts categorical columns into numeric values"""
        numeric_data = []
        cols = len(self.header)

        # Determine which columns are numeric and which are categorical
        is_numeric = [True] * cols
        for row in self.data:
            for i in range(cols):
                try:
                    float(row[i])
                except ValueError:
                    is_numeric[i] = False

        # Create encoders for categorical columns
        for i in range(cols):
            if not is_numeric[i]:
                col_name = self.header[i]
                self.encoders[col_name] = {}
                next_code = 0.0
                for row in self.data:
                    value = row[i].strip().lower()
                    if value not in self.encoders[col_name]:
                        self.encoders[col_name][value] = next_code
                        next_code += 1.0

        # Build numeric data array
        for row in self.data:
            new_row = []
            for i in range(cols):
                value = row[i].strip().lower()
                if is_numeric[i]:
                    new_row.append(float(value))
                else:
                    col_name = self.header[i]
                    new_row.append(self.encoders[col_name][value])
            numeric_data.append(new_row)

        return np.array(numeric_data, dtype=float)

    def split_features_labels(self, target_column: str):
        """Splits dataset into X (features) and y (labels)"""
        target_column = target_column.strip().lower()  # normalize input
        if target_column not in self.header:
            raise ValueError(
                f"Column '{target_column}' not found in CSV headers: {self.header}"
            )

        target_index = self.header.index(target_column)
        numeric_data = self.encode_data()
        y = numeric_data[:, target_index]
        X = np.delete(numeric_data, target_index, axis=1)
        return X, y


def main():
    cleaner = CSVCleaner("../datasets/housing.csv")
    cleaner.load_csv()
    print(f"CSV File: {cleaner.filename}")
    print(f"Columns: {cleaner.header}")

    numeric_data = cleaner.encode_data()
    print("\n=== Cleaned Numeric Data ===")
    print(numeric_data)

    print("\n=== Encoded Value Mappings ===")
    print(cleaner.encoders)

    # Automatically use lowercase 'price' as target
    X, y = cleaner.split_features_labels("price")
    print("\n=== Features (X) Shape ===", X.shape)
    print("=== Labels (y) Shape ===", y.shape)


if __name__ == "__main__":
    main()
