#!/bin/bash
for i in {1..120}
do
    echo "running $i"
    python hsmmHPC.py $i
done

