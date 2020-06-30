#!/usr/bin/env bash
export PATH=${PATH}:/meshlab/src/distrib

function meshlabserver() { xvfb-run -a -s "-screen 0 800x600x24" meshlabserver $@; }
export -f meshlabserver

#jupyter lab "$@" --allow-root
jupyter notebook --allow-root
