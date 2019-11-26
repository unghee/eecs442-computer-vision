#!/usr/bin/python
#David Fouhey
#442 submission format checker
#
#This should only accept a zip file containing a single folder ${uniqname}
#In particular, it will fail for:
#   -Files outside that folder
#   -Multiple folders (although __MACOSX is fine)
#   -Non-zip files

import zipfile, os, sys

def die(s):
    print("%s\nExiting Unsuccessfully." % s)
    sys.exit(1)

def firstFolder(path):
    """Return the first folder in the path"""
    while path != "":
        nextPath, _ = os.path.split(path)
        if nextPath == "":
            return path
        path = nextPath

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("%s zipname" % (sys.argv[0]))
        sys.exit(1)

    filen = sys.argv[1]

    if not os.path.exists(filen):
        die("Oops! %s doesn't exist" % filen)

    if not zipfile.is_zipfile(filen):
        die("Oops! %s is not a zipfile" % filen)
        
    try:
        zf = zipfile.ZipFile(filen,'r')
    except:
        die("Oops! I can't can't open %s")

    subdirs = set([])

    for zfFilen in zf.namelist():
        head, tail = os.path.split(zfFilen)
        if head == "":
            die("I found a file that's not in a directory: %s" % zfFilen)
        subdirs.add(firstFolder(zfFilen))

    #handle macs, sigh
    subdirs.discard("__MACOSX")

    if len(subdirs) > 1:
        die("There are multiple root subfolders: %s" % (", ".join(list(subdirs))))

    zf.close()
    print("Tests passsed, assuming your uniqname is ``%s''" % list(subdirs)[0])
    

