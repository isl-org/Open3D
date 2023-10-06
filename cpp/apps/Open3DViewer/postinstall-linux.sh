#!/bin/sh

if [ $(id -u) = 0 ]; then
    update-mime-database /usr/share/mime # add new MIME types
    update-desktop-database # associate MIME -> app
    gtk-update-icon-cache /usr/share/icons/hicolor || true
else
    update-mime-database ~/.local/share/mime # add new MIME types
    update-desktop-database 2> /dev/null # associate MIME -> app
                                         # (junk confusing errors)
    # Desktop Environments seem to scan ~/.local for icons so no need for
    # gtk-update-icon-cache.
fi
