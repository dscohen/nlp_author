import glob
import xml.etree.ElementTree as ET

class Provider:
    def get(self):
        data_path = "slate/data/"

        filename_to_author = {}
        authors_idx = {}

        for anc in glob.glob(data_path + "*.anc"):
            root = ET.parse(anc).getroot()
            filename = root.find('{http://www.xces.org/ns/GrAF/1.0/}profileDesc').find('{http://www.xces.org/ns/GrAF/1.0/}primaryData').attrib['loc']
            author = root.find('{http://www.xces.org/ns/GrAF/1.0/}fileDesc') \
                         .find('{http://www.xces.org/ns/GrAF/1.0/}sourceDesc') \
                         .find("{http://www.xces.org/ns/GrAF/1.0/}author")
            if author != None:
                filename_to_author[filename] = author.text
                if author.text not in authors_idx: authors_idx[author.text] = len(authors_idx)

        data = []
        tags = []

        for txt in glob.glob(data_path + "*.txt"):
            filename = txt[len(data_path):]
            if filename in filename_to_author:
                author_idx = authors_idx[filename_to_author[filename]]
                text = open(txt).read()
                data.append(text)
                tags.append(author_idx)

        idx_to_author = {v: k for k, v in authors_idx.iteritems()}

        return {"data": data, "tags": tags, "idx": idx_to_author}