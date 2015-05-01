import csv



class rest_id:
    def __init__(self,rest_id):
        self.rest_id = rest_id
        self.f_dates = []
        self.f_stars = []
        self.y_star = []
        self.reviews = []
        self.rev_dates = []

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

def get_yelp_id():
    '''returns dict of yelp_id -> rest_id class'''
    yelp_rest = {}
    tr = {}
    with open("data/restaurant_ids_to_yelp_ids.csv") as f:
        reader = csv.reader(f)
        for row in reader:
            yelp_rest[row[1]] = rest_id(row[0])
            tr[row[0]] = row[1]
    with open("data/train_labels.csv") as f:
        reader = csv.reader(f)
        for row in reader:
               yelp_rest[tr[row[2]]].f_stars.append([row[3],row[4],row[5]])
               yelp_rest[tr[row[2]]].f_dates.append(row[1])
    with open("data/yelp_academic_dataset_review.csv") as f:
        reader = csv.reader(f)
        for row in reader:
            if row[4] in yelp_rest:
                yelp_rest[row[4]].reviews.append(row[2])
                yelp_rest[row[4]].rev_dates.append(row[7])
                yelp_rest[row[4]].y_star.append(row[6])

    return yelp_rest

# def read_reviews(yelp_to_b):
#     with open("data/yelp_academic_dataset_review.json") as f:
#         data = json.loads(f.read())

if __name__ == "__main__":
    yelp_to_b = read_yelp_to_b()
    targets = read_targets()
    objects =  get_yelp_id()
    #reviews = read_reviews(yelp_to_b)
