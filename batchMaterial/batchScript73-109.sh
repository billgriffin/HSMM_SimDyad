#!/bin/bash

for i in {73..109}
do
    echo "running $i"
    python hsmmHPC.py $i
done
