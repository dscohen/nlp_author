import csv



class Restaurant:
    def __init__(self, rest_id):
        self.rest_id = rest_id
        self.f_dates = []
        self.f_stars = []
        self.y_star = []
        self.reviews = []
        self.rev_dates = []

def get_yelp_id():
    '''returns dict of yelp_id -> Restaurant class'''
    yelp_rest = {}
    tr = {}
    with open("yelp/data/restaurant_ids_to_yelp_ids.csv") as f:
        reader = csv.reader(f)
        header = reader.next()
        for row in reader:
            yelp_rest[row[1]] = Restaurant(row[0])
            tr[row[0]] = row[1]
    with open("yelp/data/train_labels.csv") as f:
        reader = csv.reader(f)
        header = reader.next()
        for row in reader:
            yelp_rest[tr[row[2]]].f_stars.append([int(row[3]), int(row[4]), int(row[5])])
            yelp_rest[tr[row[2]]].f_dates.append(row[1])
    with open("yelp/data/yelp_academic_dataset_review.csv") as f:
        reader = csv.reader(f)
        header = reader.next()
        for row in reader:
            if row[4] in yelp_rest:
                yelp_rest[row[4]].reviews.append(row[2])
                yelp_rest[row[4]].rev_dates.append(row[7])
                yelp_rest[row[4]].y_star.append(int(row[6]))

    return yelp_rest

def get_data():
    "Build a list of tuples of the form (features, tag) for the Yelp data."
    restaurants = get_yelp_id()
    result = []
    for rest_id, r in restaurants.iteritems():
        if len(r.f_stars) > 0:
            # Possible features:
            # Average yelp rating, concatenation of all reviews, average of previous inspection grades
            # Count of "dirty" and/or "clean" words in reviews and/or tags
            avg_rating = sum(r.y_star) / float(len(r.y_star))
            yelp_reviews = "\n".join(r.reviews)
            grades = [map(lambda x: x[i], r.f_stars) for i in range(3)]  # list of lists of number of stars
            if len(grades[0]) == 0 and len(grades[1]) == 0 and len(grades[2]) == 0:
                avg_grades = [0, 0, 0]
            else:
                avg_grades = [sum(l)/float(len(l)) for l in grades]            

            features = {
                "avg_rating": avg_rating,
                "reviews": yelp_reviews,
                "avg_one_star_grades": avg_grades[0],
                "avg_two_star_grades": avg_grades[1],
                "avg_three_star_grades": avg_grades[2]
            }
            tag = r.f_stars[-1] # Use the last grade as the tag
            result.append((features, tag))
    return result

if __name__ == "__main__":
    yelp_to_b = read_yelp_to_b()
    targets = read_targets()
    objects = get_yelp_id()
    print objects["NOxXooahnQvpHoYKYTdh8g"].reviews
