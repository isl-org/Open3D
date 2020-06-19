#source from /bin/bash
modifyInclude() {
	perl -pe "s{^(#include .)$1/}{\$1$2/}"
}

mvDir() {
	sd=$1
	td=$2
	[ -e $sd ]||{ echo "$sd doesn't exist";return;}
	[ -e $td ]&&{ echo "$td already exist";return;}
	git mv "$sd" "$td" ||{ echo "failed mv";return;}
	echo "$1 $2" >> reorg/movedDirs.lst
}

#moves file/dir to new place and records the change
#if you want git to figure out that the move happened, commit and then run applyAllIncludeChanges
#this way if the files are unchanged git heuristics will work for detecting file movement
mvInclude() {
	sd=cpp/$1
	td=cpp/$2
	[ -e $sd ]||{ echo "$sd doesn't exist";return;}
	[ -e $td ]&&{ echo "$td already exist";return;}
	git mv "$sd" "$td" ||{ echo "failed mv";return;}
	echo "$1 $2" >> reorg/applyIncludes.lst
}

applySingleIncludeChange() {
	find cpp examples -name '*.[ch]*'|while read f
	do
		cat $f |
		modifyInclude "$1" "$2" |
		cat > $f.reorgincl
		diff --brief $f $f.reorgincl > /dev/null || echo $f
		#preserve permissions
		cat $f.reorgincl > $f
		rm $f.reorgincl
	done
	echo "$1 $2" >> reorg/movedIncludes.lst
}

applyAllIncludeChanges() {
	cat reorg/applyIncludes.lst|while read l
	do
		applySingleIncludeChange $l
	done
	rm reorg/applyIncludes.lst
}

#apply a perl change to all (most) files (exclude 3rdparty, builds, reorg, and some others)
#$1 is a perl command to modify a line, usually would be s/// as in: s/unit_test/tests/g but can be other perl commands
perlAllFiles() {
	find CHANGELOG.md CMakeLists.txt cpp docs examples python README.md util -type f|while read f
	do
		cat $f |
		perl -pe "$1" |
		cat > $f.perlch
		diff --brief $f $f.perlch > /dev/null || echo $f
		#preserve permissions
		cat $f.perlch > $f
		rm $f.perlch
	done
	echo "$1" >> reorg/perlChanges.lst
}
