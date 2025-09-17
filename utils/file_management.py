import csv


def load_from_csv(filename):
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        return [[int(y) for y in row] for row in reader]

def load_first_n_lines(pool_size, input_file):
    res = []
    with open(input_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i >= pool_size:
                break
            res.append(row)
    return res

def save_to_csv(filename, data, mode):
    """
    Save data to a CSV file.

    Parameters:
    filename (str): The name of the file to save the data to.
    data (list of lists): The data to be written to the CSV file. Each inner list represents a row.
    mode (str): The mode in which to open the file. Must be either 'w' for write or 'a' for append.

    Raises:
    ValueError: If the mode is not 'w' or 'a'.

    Example:
    >>> save_to_csv('output.csv', [['Name', 'Age'], ['Alice', 30], ['Bob', 25]], 'w')
    """
    if mode not in ['w', 'a']:
        raise ValueError('mode must be either "w" or "a"')
    with open(filename, mode = mode, newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)