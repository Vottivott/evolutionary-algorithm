import sys
if len(sys.argv) == 2:
    drivevariant = str(sys.argv[1])
    del sys.argv[1]
else:
    drivevariant = None

from src.evoant.worm_simulation import *
main(drivevariant)
