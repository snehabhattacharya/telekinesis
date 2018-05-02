import urllib2
import os

filepath = "/home/snehabhattac/ubicompdata/S"
path = "https://www.physionet.org/pn4/eegmmidb/S"
for i in range(1,29):
  f = '{0:03}'.format(i)
  for j in range(1,15):
      g = '{0:02}'.format(j)
      url = path + f +"/S"+f +"R"+g + ".edf"
      print url
      response = urllib2.urlopen(url)
      filename = "S"+g+".edf"
      file_ = open(filepath+str(i)+"/" + filename, 'w')
      file_.write(response.read())
      file_.close()

# for i in range(1,110):
#     f = '{0:03}'.format(i)
#     url = path + f +"/S"+f +"R"+"14" + ".edf"
#     print url
#     response = urllib2.urlopen(url)
#     filename = "S14.edf"
#     file_ = open(filepath+str(i)+"/" + filename, 'w')
#     file_.write(response.read())
#     file_.close()

    
