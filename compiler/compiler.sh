#!/bin/sh

curl -H "Accept: application/vnd.github.v3+json" -L https://api.github.com/repos/google-coral/edgetpu/tarball/master | \
  tar -xz --strip=2 google-coral-edgetpu-6d69966/compiler/x86_64/
sudo mv x86_64/* /usr/local/bin
edgetpu_compiler -v