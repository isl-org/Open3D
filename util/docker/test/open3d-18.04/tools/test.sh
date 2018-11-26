#!/bin/sh

. ./name.sh
./stop.sh

# run container with the shared folder as a bind mount
docker container run \
       --rm \
       -d \
       -p 5920:5900 \
       -h $NAME \
       --name $NAME \
       $NAME

# docker container exec -it -w $Open3D_DOCK $NAME bash -c 'git clone https://github.com/IntelVCL/Open3D.git open3d && cd open3d && mkdir -p build && cd build && cmake .. -DCMAKE_INSTALL_PREFIX=~/open3d_install -DBUILD_UNIT_TESTS=ON && make -j && make install && bash'
# docker container exec -it -w $Open3D_DOCK $NAME bash -c 'build.sh && bash'
# docker container exec -it -w $Open3D_DOCK $NAME bash -c '/root/open3d/util/scripts/install-gtest.sh && bash'

# docker container exec -it -w $Open3D_DOCK $NAME bash -c 'wget https://github.com/google/googletest/archive/release-1.8.0.tar.gz -O /tmp/release-1.8.0.tar.gz && \
#                                                          cd /tmp/ && \
#                                                          tar -xzvf /tmp/release-1.8.0.tar.gz && \
#                                                          cd /tmp/googletest-release-1.8.0 && \
#                                                          mkdir build && \
#                                                          cd build && \
#                                                          cmake .. && \
#                                                          make -j && \
#                                                          cd googlemock/gtest && \
#                                                          cp lib*.a /usr/local/lib && \
#                                                          cd ../../../googletest && \
#                                                          cp -r include/gtest /usr/local/include/gtest && \
#                                                          cd ../.. && \
#                                                          bash'

# Open3D_DOCK=/root/open3d
# docker container exec -it -w $Open3D_DOCK $NAME bash -c './build.sh && bash'

Open3D_DOCK=/root
docker container exec -it -w $Open3D_DOCK $NAME bash -c 'git clone https://github.com/IntelVCL/Open3D.git open3d && \
                                                         cd open3d && \
                                                         mkdir -p build && \
                                                         cd build && \
                                                         cmake .. -DCMAKE_INSTALL_PREFIX=~/open3d_install -DBUILD_UNIT_TESTS=ON -DCMAKE_BUILD_TYPE=Release && \
                                                         echo && \
                                                         make -j && \
                                                         echo && \
                                                         make install && \
                                                         bash'

                                                      #  make install-pip-package && \
