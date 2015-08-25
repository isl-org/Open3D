GLEW - The OpenGL Extension Wrangler Library

       http://glew.sourceforge.net/
       https://github.com/nigels-com/glew

See doc/index.html for more information.

If you downloaded the tarball from the GLEW website, you just need to:

    Unix:

        make

        sudo -s

        make install

        make clean

    Windows:

        use the project file in build/vc12/

If you wish to build GLEW from scratch (update the extension data from
the net or add your own extension information), you need a Unix
environment (including wget, perl, and GNU make).  The extension data
is regenerated from the top level source directory with:

        make extensions

An alternative to generating the GLEW sources from scratch is to
download a pre-generated (unsupported) snapshot:

        https://sourceforge.net/projects/glew/files/glew/snapshots/
