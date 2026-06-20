#!/bin/bash
g++ src/main.cpp src/canny.cpp -Isrc -Iinclude -o canny_exec
./canny_exec
