import csv
import json

def read_yelp_to_b():
    yelp_to_b = {}
    with open("data/restaurant_ids_to_yelp_ids.csv") as f:
        reader = csv.reader(f)
        headers = reader.next()
        for row in reader:
            for i in range(1, len(row)):
                if row[i]: yelp_to_b[row[i]] = row[i]
    return yelp_to_b

def read_targets():
    targets = {}
    with open("data/train_labels.csv") as f:
        reader = csv.reader(f)
        headers = reader.next()
        for row in reader:
            if row[2] not in targets: targets[row[2]] = {}
            targets[row[2]][row[1]] = row[3:]
    return targets

# def read_reviews(yelp_to_b):
#     with open("data/yelp_academic_dataset_review.json") as f:
#         data = json.loads(f.read())

if __name__ == "__main__":
    yelp_to_b = read_yelp_to_b()
    targets = read_targets()
    #reviews = read_reviews(yelp_to_b)
