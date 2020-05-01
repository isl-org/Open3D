#/bin/bash

if [ $(id -u) = 0 ]; then
    update-mime-database /usr/share/mime # add new MIME types
    update-desktop-database # associate MIME -> app
else
    update-mime-database ~/.local/share/mime # add new MIME types
    update-desktop-database 2> /dev/null # associate MIME -> app
                                         # (junk confusing errors)
fi
