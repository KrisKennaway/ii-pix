#!/bin/sh

APPLECOMMANDER=~/bin/AppleCommander.jar

for i in examples/*-original.png; do
  python convert.py --lookahead 8 --palette ntsc --dither floyd $i $(echo $i | sed -e 's,original.png,iipix-ntsc.dhr,')
  python convert.py --lookahead 8 --palette virtualii --dither jarvis-mod --no-show-output $i $(echo $i | sed -e 's,original.png,iipix-virtualii.dhr,')
  python convert.py --lookahead 8 --palette openemulator --dither jarvis-mod  --no-show-output $i $(echo $i | sed -e 's,original.png,iipix-openemulator.dhr,')
done

rm -f examples/examples.po
java -jar $APPLECOMMANDER -pro800 examples/examples.po NTSC
idx=0
for i in examples/*-iipix-ntsc.dhr; do
  idx=$((idx+1))
  java -jar $APPLECOMMANDER -p examples/examples.po ntsc.$idx BIN 0x2000 < $i
done

idx=0
for i in examples/*-iipix-openemulator.dhr; do
  idx=$((idx+1))
  java -jar $APPLECOMMANDER -p examples/examples.po openem.$idx BIN 0x2000 < $i
done