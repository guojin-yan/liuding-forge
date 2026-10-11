#!/bin/bash

# Format all C++ source files

find include src tests examples -name "*.hpp" -o -name "*.cpp" | xargs clang-format -i

echo "Code formatting complete!"
