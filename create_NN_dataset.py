from get_NN_dataset import generate_dataset


filename = "test.pt"
dataset = generate_dataset(10_000, 5, filename)
print(dataset.__len__())