#!/bin/sh

# function that will exit the script if not enough arguments.
die () {
    echo >&2 "$@"
    exit 1
}


# the arguments specify which files and folders to purge.
[ "$#" -eq 0 ] && die "at least 1 argument required, $# provided"

rm_args="$@"

# du tells us about the current size of the .git folder.
du -s .git

# based on http://stackoverflow.com/questions/2100907/how-do-i-purge-a-huge-file-from-commits-in-git-history
# with added stuff to make it work for multiple files and directories.
git filter-branch -f --prune-empty -d /dev/shm/scratch \
 --index-filter "git rm -rf --cached --ignore-unmatch $rm_args" \
 --tag-name-filter cat -- --all

du -s .git

# prepare garbage collection
git reflog expire --expire=now --all

du -s .git

# garbage collection
git gc --prune=now

# have a look at final file size.
du -s .git