#!/usr/bin/env bash
set -e

# Open3D's Python Tutorial require following packages
# - numpy
# - matplotlib
# - opencv

# To install these package, try one of the following
while true; do
	read -p "Install Python dependencies using pip or conda? (p: pip, c: conda, q: quit) : " pcq
		case $pcq in
			[Pp]* )
			# for pip users
			python -m pip install --user numpy matplotlib opencv-python;
			break;;

			[Cc]* )
			# for anaconda users
			conda create --name py36_open3d python=3.6
			source activate py36_open3d
			conda install -y numpy matplotlib
			conda install -y -c conda-forge opencv
			break;;

			[Qq]* )
			# quit
			exit;;

			* )
			echo "Please answer p or c.";;
		esac
done
