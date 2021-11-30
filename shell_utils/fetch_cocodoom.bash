#! /bin/bash

pushd "$(pwd)" || exit 1

cd "$1" || exit 1

if [ -d cocodoom ]
then
  echo " [Fetch COCODoom] - cocodoom directory already exists"
  exit 0
fi

wget https://www.robots.ox.ac.uk/~vgg/share/cocodoom-v1.0.tar.gz
tar xzf cocodoom-v1.0.tar.gz
rm cocodoom-v1.0.tar.gz

popd || exit
