cmake -G "Visual Studio 16 2019" -T host=x64 -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=./install .
cmake --build . --config Release --target install
pause