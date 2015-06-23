#!/bin/bash

for i in {1..36}
do
    echo "running $i"
    python hsmmHPC.py $i
done
