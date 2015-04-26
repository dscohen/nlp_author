import csv

class Provider:
    def get(self):
        idx  = ["M", "F"]
        idx_map = {"M": 0, "F": 1}
        data = []
        tags = []

        with open("blog/data.csv", "rU") as f:
            f = csv.reader(f)
            for row in f:
                if (row[1] == "M" or row[1] == "F"):
                    try:    # Can get an encoding error. So, just ignore it.
                        data.append(row[0].decode("utf-8"))
                        tags.append(idx_map[row[1]])
                    except:
                        pass

        return {"data": data, "tags": tags, "idx": idx}