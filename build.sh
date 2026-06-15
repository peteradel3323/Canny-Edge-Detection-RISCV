#!/bin/bash
g++ src/main.cpp src/canny.cpp -Isrc -o canny_exec
./canny_exec