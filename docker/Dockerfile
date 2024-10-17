ARG IDF_TAG
FROM docker.io/espressif/idf:$IDF_TAG
ARG UNAME="nonroot"
ARG UID="1000"
ARG GID="1000"


# Add non-root user.
RUN echo "$GID"
RUN groupadd -g $GID -o $UNAME
RUN useradd -m -u $UID -g $GID -G $UNAME -o -s /bin/bash $UNAME
RUN echo '. /opt/esp/idf/export.sh' >> "/home/$UNAME/.bashrc"
RUN chown -R $UNAME /opt/esp/idf/

# Dependencies for linux building
RUN apt-get update && apt-get install libsodium-dev libmbedtls-dev -y

# GCC 12, also linux building
RUN wget https://ftpmirror.gnu.org/gcc/gcc-12.3.0/gcc-12.3.0.tar.xz &&    tar xf gcc-12.3.0.tar.xz &&    cd gcc-12.3.0 &&    contrib/download_prerequisites
RUN mkdir build && \
    cd build && \
    ../gcc-12.3.0/configure -v --build=x86_64-linux-gnu --host=x86_64-linux-gnu --target=x86_64-linux-gnu  \
    --prefix=/usr/local/gcc-12.3.0 --enable-checking=release --enable-languages=c,c++,fortran --disable-multilib  \
    --program-suffix=-12.3 && \
    make -j 8 && \
    make install-strip

RUN echo 'PATH="/usr/local/gcc-12.3.0/bin:$PATH"' >> "/etc/profile"
RUN echo '/usr/local/gcc-12.3.0/lib64/' >> '/etc/ld.so.conf.d/gcc12.conf'
RUN ldconfig

# install protoc compiler
RUN curl -LO https://github.com/protocolbuffers/protobuf/releases/download/v3.20.3/protoc-3.20.3-linux-x86_64.zip && \
    unzip protoc-3.20.3-linux-x86_64.zip -d /tmp/protoc && \
    mv /tmp/protoc/bin/protoc /usr/local/bin/protoc && \
    mv /tmp/protoc/include/google /usr/local/include && \
    rm -rf /tmp/protoc protoc-3.20.3-linux-x86_64.zip

# grant dialout to user
RUN usermod -a -G dialout $UNAME

USER $UNAME

