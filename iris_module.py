# imports
import os
import random
import csv
import urllib.request
import zipfile

# directories and URL for files download and transform
ROOT_DIR = os.getcwd()
EXTRACT_PATH = os.path.join(ROOT_DIR, 'data')
OUTPUT_PATH = os.path.join(ROOT_DIR, 'output_data')
URL_PATH = "https://archive.ics.uci.edu/static/public/53/iris.zip"

# check if directories exist, if not create
os.makedirs(EXTRACT_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)


class DataProcessor:
    @staticmethod
    def fetch_data(extract_path=EXTRACT_PATH, url=URL_PATH):
        """
        Fetches the Iris Plants database from a specified URL and extracts it to the provided directory.

        Args:
        - extract_path (str, optional): The path to extract the downloaded dataset. Defaults to EXTRACT_PATH.
        - url (str, optional): The URL from which to download the Iris Plants database. Defaults to URL_PATH.

        Returns:
        None

        This function downloads the Iris Plants database from the given URL and extracts it to the specified directory.
        If the directory specified by `extract_path` does not exist, it creates the directory before downloading.
        """

        if not os.path.isdir(extract_path):
            os.makedirs(extract_path)

        zip_path = os.path.join(extract_path, "iris.zip")
        urllib.request.urlretrieve(url, zip_path)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(path=extract_path)
        print("Pobrano i rozpakowano bazę danych Iris Plants do folderu 'DATA'\n")

    @staticmethod
    def read_csv(header=False):
        """
        Reads the Iris Plants dataset stored in a CSV file.

        Args:
        - header (bool, optional): Specifies if the CSV file contains a header. Defaults to False.

        Returns:
        tuple: A tuple containing X, y, and column names.

        This static method reads the Iris Plants dataset stored in a CSV file located at the specified path.
        It reads the features (X) and labels (y) from the CSV file. If `header` is True, it assumes
        the first row contains column names and separates them accordingly.

        Returns the extracted features (X) and labels (y) as lists. If `header` is True, it also returns
        the column names as a list. The features are extracted from the first four columns and the labels
        from the fifth column of the CSV file.
        """

        path = os.path.join(EXTRACT_PATH, "iris.data")
        X = []
        y = []
        col_names = []

        with open(path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            if header:
                col_names = next(reader)  # Read column names if header is True

            for row in reader:
                if len(row) != 0:  # check if row contains the data
                    X.append(row[:4])  # Nested lists for X data
                    y.extend(row[4:])  # y as one-dimensional list

        return X, y, col_names

    @staticmethod
    def show_labels(labels=None):
        """
        Displays data labels.

        Parameters:
        labels (list): List of labels.

        Returns:
        None
        """

        if not labels:
            print("Brak etykiet w pliku: \n", labels)
        else:
            print("Etykiety kolumn: \n", labels)

    @staticmethod
    def disp_data_slice(X, y, start_idx=None, stop_idx=None):
        """
        Displays dataset labels and data within the specified range of indices.

        Parameters:
        X (list): Data.
        y (list): Labels.
        start_idx (int, optional): Starting index of the range to display. Default is None.
        stop_idx (int, optional): Ending index (exclusive) of the range to display. Default is None.

        Returns:
        None
        """

        if start_idx is None and stop_idx is None:
            print("\nEtykiety i dane:")
            for i in range(len(y)):
                print(y[i], X[i])
        else:
            print("\nEtykiety i dane dla wybranego zakresu:")
            for i in range(start_idx, stop_idx):
                print(y[i], X[i])

    @staticmethod
    def train_test_split(X, y, train_size=0.7, val_size=0.1, test_size=0.2):
        """
        Splits the dataset into training, validation, and test sets based on provided proportions.

        Parameters:
        X (list): Data.
        y (list): Labels.
        train_size (float, optional): The proportion of the dataset to include in the training set. Default is 0.7.
        val_size (float, optional): The proportion of the dataset to include in the validation set. Default is 0.1.
        test_size (float, optional): The proportion of the dataset to include in the test set. Default is 0.2.

        Returns:
        tuple: A tuple containing the split datasets - X_train, X_val, X_test, y_train, y_val, y_test.
        """

        # Check if provided proportions sum up to 1
        if (train_size + val_size + test_size) != 1:
            raise ValueError("Wartości train_size, val_size i test_size muszą się sumować do 1!")

        # shuffle the data
        random.seed(42)  # set seed for reproducibility
        combined_data = list(zip(X, y))  # merge labels and data for shuffle
        random.shuffle(combined_data)
        X, y = zip(*combined_data)  # unpack the data

        # calculate split indices
        train_idx = round(train_size * len(y))
        val_idx = round((train_size + val_size) * len(y))

        # split the data
        X_train, X_val, X_test = X[:train_idx], X[train_idx:val_idx], X[val_idx:]
        y_train, y_val, y_test = y[:train_idx], y[train_idx:val_idx], y[val_idx:]

        return X_train, X_val, X_test, y_train, y_val, y_test

    @staticmethod
    def disp_label_cardinality(y):
        """
        Display cardinality of unique classes.

        Parameters:
        y (list): A list containing class labels.

        Returns:
        list: A list of tuples containing unique class labels and their respective cardinality.
        """

        unique_labels = list(set(y))
        print("\nKlasy i ich liczebności w zbiorze danych: ")
        [print((label, cardinality)) for cardinality, label in enumerate(unique_labels)]

    @staticmethod
    def disp_selected_label_data(X, y, selected_label="Iris-setosa"):
        """
        Display the data of selected label.

        Args:
        - y (list): List of labels.
        - selected_label (str, optional): The label for which associated data will be displayed.
                                     Defaults to "Iris-setosa".

        Example:
        To display data associated with the label "Iris-versicolor":
        ```
        disp_selected_label_data(y=y, selected_label="Iris-versicolor")
        ```
        """
        combined_data = list(zip(y, X))
        print("\nDane dla wybranej klasy: ")
        [print(i) for i in combined_data if i[0] == selected_label]


class DataSaver:
    @staticmethod
    def save_as_csv(data, file_name):
        """
        Save a prepared dataset as a CSV file.

        Parameters:
        data (list): The list to be saved as a CSV file.
        file_name (str): The name of the CSV file.

        Returns:
        None
        """
        path = os.path.join(OUTPUT_PATH, file_name)
        with open(path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)
        print(f'Data saved as {file_name}')


def main():
    DataProcessor.fetch_data()  # download and unpack dataset
    X, y, labels = DataProcessor.read_csv(header=False)  # read data to python lists
    DataProcessor.show_labels(labels)  # example for showing labels if they are present in the first row
    DataProcessor.disp_data_slice(X, y, 0, 4)  # example of displaying sliced dataset

    # train, test, val split
    X_train, X_val, X_test, y_train, y_val, y_test = DataProcessor.train_test_split(X=X,
                                                                                    y=y,
                                                                                    train_size=0.7,
                                                                                    val_size=0.1,
                                                                                    test_size=0.2)
    DataProcessor.disp_label_cardinality(y)  # display label counts

    # Display the selected label from the chosen dataset (this one is on test dataset)
    DataProcessor.disp_selected_label_data(X_test, y_test, selected_label="Iris-versicolor")

    # defined list of tuples (<dataset_list>, <name_of_file_as_str) for saving function
    save_list = [
        (X, "X_full.csv"),
        (y, "y_full.csv"),
        (X_train, "X_train.csv"),
        (y_train, "y_train.csv"),
        (X_val, "X_val.csv"),
        (y_val, "y_val.csv"),
        (X_test, "X_test.csv"),
        (y_test, "y_test.csv")
    ]

    # loop over a list of datasets
    for data, file_name in save_list:
        try:
            DataSaver.save_as_csv(data, file_name)
        except Exception as e:
            print(f"Wystąpił problem podczas zapisu danych do pliku: {e}")


if __name__ == "__main__":
    main()
