import glob
import xml.etree.ElementTree as ET

data_path = "data/"

authors = {}
for anc in glob.glob(data_path + "*.anc"):
    root = ET.parse(anc).getroot()
    filename = root.find('{http://www.xces.org/ns/GrAF/1.0/}profileDesc').find('{http://www.xces.org/ns/GrAF/1.0/}primaryData').attrib['loc']
    author = root.find('{http://www.xces.org/ns/GrAF/1.0/}fileDesc') \
                 .find('{http://www.xces.org/ns/GrAF/1.0/}sourceDesc') \
                 .find("{http://www.xces.org/ns/GrAF/1.0/}author")
    if author != None:
        authors[filename] = author.text

data = {}

for txt in glob.glob(data_path + "*.txt"):
    filename = txt[len(data_path):]
    if filename in authors:
        author = authors[filename]
        text = open(txt).read()
        if author not in data: data[author] = []
        data[author].append(text)

print len(data), "authors"
print len(authors), "articles"
print float(len(authors))/len(data), "average docs per author"