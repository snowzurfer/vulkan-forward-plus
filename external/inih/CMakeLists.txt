# Auth: Alberto Taiuti

cmake_minimum_required(VERSION 2.8)

project(inih-proj)

# Export commands for YCM
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}
                    ${CMAKE_CURRENT_SOURCE_DIR}/cpp)

add_library(inih ${CMAKE_CURRENT_SOURCE_DIR}/ini.c
            ${CMAKE_CURRENT_SOURCE_DIR}/cpp/INIReader.cpp)
