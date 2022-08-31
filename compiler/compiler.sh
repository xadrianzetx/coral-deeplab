#!/bin/sh

git clone https://github.com/google-coral/edgetpu.git
sudo mv edgetpu/compiler/x86_64/* /usr/local/bin
edgetpu_compiler -v