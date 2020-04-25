#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
from os import path, system
from shlex import quote


docker_image   = 'nvcr.io/nvidia/tensorflow:20.03-tf2-py3'
root, _ = path.split(path.realpath(__file__))
parameters = quote(' '.join(sys.argv[1:]))

cmd = F'docker run -it --rm \
        -w "/workdir" \
        -v "{root}:/workdir" \
        -v "/:/host" \
        {docker_image} \
        /bin/bash -c " \
        pip install -r requirements.txt && \
        {parameters}"'

system(cmd)
