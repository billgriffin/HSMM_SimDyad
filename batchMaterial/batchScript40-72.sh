#!/bin/bash

for i in {40..72}
do
    echo "running $i"
    python hsmmHPC.py $i
done
