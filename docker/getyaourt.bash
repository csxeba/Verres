#! /bin/bash

wget https://aur.archlinux.org/cgit/aur.git/snapshot/package-query.tar.gz
wget https://aur.archlinux.org/cgit/aur.git/snapshot/yaourt.tar.gz
tar xzf package-query.tar.gz
tar xzf yaourt.tar.gz
cd package-query || exit 1
makepkg -sc
sudo pacman -U $(ls | grep package-query | grep .tar.xz) --noconfirm
cd ../yaourt || exit 1
makepkg -sc
sudo pacman -U $(ls | grep yaourt | grep .tar.xz) --noconfirm
cd ..
rm yaourt.tar.gz
rm -fr yaourt
rm package-query.tar.gz
rm -fr package-query
