import urllib2


def retrieve_information_web(url):
    try:
        res = urllib2.urlopen(url).readlines()
    except:
        res = []
    return res


def retrieve_information_file(input):
    res = []
    with open(input, "r") as f:
        res = f.readlines()
    return res
