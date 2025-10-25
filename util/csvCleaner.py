import csv
import numpy as np

class CSVPreprocessor:
    """Preprocesses a CSV file into numeric data ready for neural networks."""

    def __init__(self, filename: str):
        self.filename = filename
        self.header = []
        self.encoders = {}
        self.data = None
    
    def _is_numeric_column(self, column):
        for val in column:
            try:
                float(val)
            except ValueError:
                return False
        return True

    def _encode_categorical_column(self, column_name, column):
        mapping = {}
        encoded = []
        next_code = 0.0
        for val in column:
            if val not in mapping:
                mapping[val] = next_code
                next_code += 1.0
            encoded.append(mapping[val])
        self.encoders[column_name] = mapping
        return encoded

    def clean(self):
        with open(self.filename, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            self.header = next(reader)
            rows = [
                row for row in reader
                if all(v.strip() != "" and v.lower() != "nan" for v in row)
            ]

        columns = list(zip(*rows))
        cleaned_columns = []

        for i, col in enumerate(columns):
            col_name = self.header[i]
            if self._is_numeric_column(col):
                cleaned_columns.append([float(v) for v in col])
            else:
                cleaned_columns.append(self._encode_categorical_column(col_name, col))

        cleaned_data = [list(row) for row in zip(*cleaned_columns)]
        self.data = np.array(cleaned_data, dtype=float)
        return self.data

    def get_mappings(self):
        return self.encoders

    def get_header(self):
        return self.header

    def summary(self):
        print("CSV File:", self.filename)
        print("Columns:", self.header)
        print("Shape:", None if self.data is None else self.data.shape)
        print("Encoders:", self.encoders)

    def split_features_labels(self, target_column: str):
        """Splits cleaned data into X (features) and y (target) based on chosen column."""
        if self.data is None:
            raise ValueError("Data not cleaned yet. Run clean() before splitting.")

        if target_column not in self.header:
            raise ValueError(f"Column '{target_column}' not found in CSV headers: {self.header}")

        target_index = self.header.index(target_column)
        y = self.data[:, target_index]
        X = np.delete(self.data, target_index, axis=1)

        print(f"Target Column: '{target_column}' (Index {target_index})")
        print(f"X Shape: {X.shape}, y Shape: {y.shape}")
        return X, y


# ======= TEST SECTION =======
if __name__ == "__main__":
    processor = CSVPreprocessor("Housing.csv")
    clean_data = processor.clean()

    print("=== Cleaned Numeric Data ===")
    print(clean_data)

    print("\n=== Encoded Value Mappings ===")
    print(processor.get_mappings())

    processor.summary()

    # Example: choose a target column for prediction (e.g., "Price")
    X, y = processor.split_features_labels("Price")

    print("\n=== Features (X) ===")
    print(X[:5])  # show first 5 rows

    print("\n=== Labels (y) ===")
    print(y[:5])
