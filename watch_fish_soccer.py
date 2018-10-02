import sys
if len(sys.argv) == 2:
    drivevariant = str(sys.argv[1])
    maincomp = False
    del sys.argv[1]
elif len(sys.argv) == 3:
    drivevariant = str(sys.argv[1])
    maincomp=True
    print "MAIN"
    del sys.argv[2]
    del sys.argv[1]
else:
    drivevariant = None

from src.evoant.worm_simulation import *
main(drivevariant, maincomp)
