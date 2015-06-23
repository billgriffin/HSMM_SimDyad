#!/bin/bash
for i in {48..120}
do
    echo "running $i"
    python hsmmHPCRn3.py $i
done

