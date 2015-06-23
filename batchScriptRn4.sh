#!/bin/bash
for i in {86..120}
do
    echo "running $i"
    python hsmmHPCRn4.py $i
done

