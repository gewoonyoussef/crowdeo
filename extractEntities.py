import sys

from nerd import NERD
timed_text = open(sys.argv[1]).readlines()
timeout = 60
n = NERD ('nerd.eurecom.fr', '1sjiljisbev18jai2m23qui9h063iiir')
results = n.extract(timed_text, 'combined', timeout) 

for i,result in enumerate(results):
  for key,value in result.items():
    print key,"\t",value
  print '------------------------------'
