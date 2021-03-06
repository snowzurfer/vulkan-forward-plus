#  MIT License

#  Copyright (c) 2017 Alberto Taiuti

#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:

#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.

#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

cmake_minimum_required(VERSION 2.8)

project(vulkan-sagres)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  message(STATUS "Build mode: debug")
else()
  message(STATUS "Build mode: release")
endif (CMAKE_BUILD_TYPE STREQUAL "Debug")

# Export commands for YCM
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

# Set custom cmake modules path
set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")

# Set default cmake build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebInfo\
MinSizeRel. Default is Debug."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)

if(MSVC)
  # Force to always compile with W1
  if(CMAKE_CXX_FLAGS_DEBUG MATCHES "/W[0-4]")
    string(REGEX REPLACE "/W[0-4]" "/W1" CMAKE_CXX_FLAGS_DEBUG
           "${CMAKE_CXX_FLAGS_DEBUG}")
  else()
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /W1")
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR
   CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  # Update if necessary
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 ")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
  set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wall")
endif()

if(UNIX)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # Color error output
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fcolor-diagnostics")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11 -stdlib=libc++")
  endif()
endif()

message(STATUS "Compiler custom flags are: ${CMAKE_CXX_FLAGS}")
message(STATUS "Compiler debug flags are: ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "Compiler release flags are: ${CMAKE_CXX_FLAGS_RELEASE}")

# Handy vars
set(GLFW_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/glfw")
set(GLM_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/glm")
set(GLI_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/gli")
set(TINYOBJLOADER_SOURCE_DIR 
    "${CMAKE_CURRENT_SOURCE_DIR}/external/tinyobjloader")
set(SONGHOTIMER_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/songho-timer")
set(ASSIMP_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/assimp")
set(LODEPNG_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/lodepng")
set(INIH_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/inih")
set(EASTL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/eastl")
set(EAStdC_SOURCE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/external/eastl/test/packages/EAStdC")
set(EAAssert_SOURCE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/external/eastl/test/packages/EAAssert")
set(SHADERC_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/shaderc")
set(VKS_BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/base")
set(VKS_FPLUS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/fplus")

# Build EASTL
#set(EASTL_BUILD_TESTS ON CACHE BOOL ON)
add_subdirectory(${EASTL_SOURCE_DIR})
add_subdirectory(${EAStdC_SOURCE_DIR})
add_subdirectory(${EAAssert_SOURCE_DIR})

# Build GLFW
set(BUILD_SHARED_LIBS OFF CACHE BOOL "")
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "")
set(GLFW_BUILD_TESTS OFF CACHE BOOL "")
set(GLFW_BUILD_DOCS OFF CACHE BOOL "")
set(GLFW_INSTALL OFF CACHE BOOL "")
add_subdirectory(${GLFW_SOURCE_DIR})

# Build timer library
add_subdirectory(${SONGHOTIMER_SOURCE_DIR})

# Build assimp
#set(BUILD_SHARED_LIBS OFF CACHE BOOL "")
#option(ASSIMP_BUILD_ASSIMP_TOOLS "" OFF)
#option(ASSIMP_BUILD_TESTS "" OFF)
option(ASSIMP_BUILD_OBJ_IMPORTER "" TRUE)
add_subdirectory(${ASSIMP_SOURCE_DIR})

# Build lodepng
add_subdirectory(${LODEPNG_SOURCE_DIR})

# Build inih
add_subdirectory(${INIH_SOURCE_DIR})

# Build shaderc 
set(SHADERC_ENABLE_SHARED_CRT ON CACHE BOOL "")
set(SHADERC_SKIP_TESTS ON CACHE BOOL "")
add_subdirectory(${SHADERC_SOURCE_DIR})

find_package(Vulkan REQUIRED)

# Set include directories
include_directories(${GLFW_SOURCE_DIR}/include
                    ${GLFW_SOURCE_DIR}/deps
                    ${GLM_SOURCE_DIR}
                    ${GLI_SOURCE_DIR}
                    ${VKS_BASE_DIR}/include
                    ${VKS_FPLUS_DIR}
                    ${TINYOBJLOADER_SOURCE_DIR}
                    ${SONGHOTIMER_SOURCE_DIR}/include
                    ${ASSIMP_SOURCE_DIR}/include
                    ${LODEPNG_SOURCE_DIR}
                    ${INIH_SOURCE_DIR}
                    ${INIH_SOURCE_DIR}/cpp
                    ${EASTL_SOURCE_DIR}/include
                    ${EAStdC_SOURCE_DIR}/include
                    ${Vulkan_INCLUDE_DIRS}/src
                    ${SHADERC_SOURCE_DIR}/libshaderc/include)

# Set headers and sources; although this method to source files
# from the source tree is not recommended by cmake, it is used here 
# for simplicity's sake
set(VKS_BASE_HEADERS
  ${VKS_BASE_DIR}/include/base_system.h
  ${VKS_BASE_DIR}/include/camera_controller.h
  ${VKS_BASE_DIR}/include/camera.h
  ${VKS_BASE_DIR}/include/crc.h
  #${VKS_BASE_DIR}/include/deferred_renderer.h
  #${VKS_BASE_DIR}/include/deferred_scene.h
  ${VKS_BASE_DIR}/include/eastl_streams.h
  ${VKS_BASE_DIR}/include/framebuffer.h
  ${VKS_BASE_DIR}/include/frustum.h
  ${VKS_BASE_DIR}/include/input_manager.h
  ${VKS_BASE_DIR}/include/light.h
  ${VKS_BASE_DIR}/include/lights_manager.h
  ${VKS_BASE_DIR}/include/logger.hpp
  ${VKS_BASE_DIR}/include/log.h
  ${VKS_BASE_DIR}/include/material_constants.h
  ${VKS_BASE_DIR}/include/material.h
  ${VKS_BASE_DIR}/include/material_instance.h
  ${VKS_BASE_DIR}/include/material_manager.h
  ${VKS_BASE_DIR}/include/material_parameters.h
  ${VKS_BASE_DIR}/include/material_texture_type.h
  ${VKS_BASE_DIR}/include/mesh.h
  ${VKS_BASE_DIR}/include/model.h
  ${VKS_BASE_DIR}/include/model_manager.h
  ${VKS_BASE_DIR}/include/renderer_type.h
  #${VKS_BASE_DIR}/include/renderer.h
  ${VKS_BASE_DIR}/include/renderpass.h
  ${VKS_BASE_DIR}/include/scene.h
  ${VKS_BASE_DIR}/include/shutdown_dtor.h
  ${VKS_BASE_DIR}/include/subpass.h
  ${VKS_BASE_DIR}/include/uncopyable.h
  ${VKS_BASE_DIR}/include/vertex_setup.h
  ${VKS_BASE_DIR}/include/viewport.h
  ${VKS_BASE_DIR}/include/meshes_heap.h
  ${VKS_BASE_DIR}/include/meshes_heap_manager.h
  ${VKS_BASE_DIR}/include/vulkan_base.h
  ${VKS_BASE_DIR}/include/vulkan_buffer.h
  ${VKS_BASE_DIR}/include/vulkan_device.h
  ${VKS_BASE_DIR}/include/vulkan_image.h
  ${VKS_BASE_DIR}/include/vulkan_swapchain.h
  ${VKS_BASE_DIR}/include/vulkan_texture.h
  ${VKS_BASE_DIR}/include/vulkan_texture_manager.h
  ${VKS_BASE_DIR}/include/vulkan_tools.h
  ${VKS_BASE_DIR}/include/vulkan_uniform_buffer.h
  ${VKS_BASE_DIR}/include/vulkan_uniform_data.h)
set(VKS_BASE_SOURCES
  ${VKS_BASE_DIR}/source/base_system.cpp
  ${VKS_BASE_DIR}/source/camera_controller.cpp
  ${VKS_BASE_DIR}/source/camera.cpp
  ${VKS_BASE_DIR}/source/crc.cpp
  #${VKS_BASE_DIR}/source/deferred_renderer.cpp
  #${VKS_BASE_DIR}/source/deferred_scene.cpp
  ${VKS_BASE_DIR}/source/eastl_opnew.cpp
  ${VKS_BASE_DIR}/source/eastl_streams.cpp
  ${VKS_BASE_DIR}/source/eastl_strings.cpp
  ${VKS_BASE_DIR}/source/framebuffer.cpp
  ${VKS_BASE_DIR}/source/frustum.cpp
  ${VKS_BASE_DIR}/source/input_manager.cpp
  ${VKS_BASE_DIR}/source/lights_manager.cpp
  ${VKS_BASE_DIR}/source/log.cpp
  #${VKS_BASE_DIR}/source/main.cpp
  ${VKS_BASE_DIR}/source/material_constants.cpp
  ${VKS_BASE_DIR}/source/material.cpp
  ${VKS_BASE_DIR}/source/material_instance.cpp
  ${VKS_BASE_DIR}/source/material_manager.cpp
  ${VKS_BASE_DIR}/source/material_parameters.cpp
  ${VKS_BASE_DIR}/source/mesh.cpp
  ${VKS_BASE_DIR}/source/model.cpp
  ${VKS_BASE_DIR}/source/model_manager.cpp
  #${VKS_BASE_DIR}/source/renderer.cpp
  ${VKS_BASE_DIR}/source/renderpass.cpp
  ${VKS_BASE_DIR}/source/scene.cpp
  ${VKS_BASE_DIR}/source/shutdown_dtor.cpp
  ${VKS_BASE_DIR}/source/subpass.cpp
  ${VKS_BASE_DIR}/source/meshes_heap.cpp
  ${VKS_BASE_DIR}/source/meshes_heap_manager.cpp
  ${VKS_BASE_DIR}/source/vertex_setup.cpp
  ${VKS_BASE_DIR}/source/vulkan_base.cpp
  ${VKS_BASE_DIR}/source/vulkan_buffer.cpp
  ${VKS_BASE_DIR}/source/vulkan_device.cpp
  ${VKS_BASE_DIR}/source/vulkan_image.cpp
  ${VKS_BASE_DIR}/source/vulkan_swapchain.cpp
  ${VKS_BASE_DIR}/source/vulkan_texture.cpp
  ${VKS_BASE_DIR}/source/vulkan_texture_manager.cpp
  ${VKS_BASE_DIR}/source/vulkan_tools.cpp
  ${VKS_BASE_DIR}/source/vulkan_uniform_data.cpp)

set(VKS_FPLUS_HEADERS
  ${VKS_FPLUS_DIR}/fplus_scene.h
  ${VKS_FPLUS_DIR}/fplus_renderer.h)
set(VKS_FPLUS_SOURCES
  ${VKS_FPLUS_DIR}/fplus_scene.cpp
  ${VKS_FPLUS_DIR}/fplus_renderer.cpp
  ${VKS_FPLUS_DIR}/main.cpp)

# Create shared library
add_library(vksagres
  ${VKS_BASE_HEADERS}
  ${VKS_BASE_SOURCES})

# Create the executables
add_executable(vksagres-fplus
  ${VKS_FPLUS_HEADERS}
  ${VKS_FPLUS_SOURCES})

# Link libraries to it
target_link_libraries(vksagres
  ${Vulkan_LIBRARIES}
  glfw
  ${GLFW_LIBRARIES}
  ${SONGHO_LIBRARY}
  ${LODEPNG_LIBRARY}
  EAStdC
  assimp
  EASTL
  inih
  shaderc)
target_link_libraries(vksagres-fplus
  vksagres)

# Set folder for assets
set(ASSETS_FOLDER ${CMAKE_CURRENT_SOURCE_DIR}/assets/)
target_compile_definitions(vksagres-fplus
  PUBLIC ASSETS_FOLDER=${ASSETS_FOLDER})
target_compile_definitions(vksagres
  PUBLIC ASSETS_FOLDER=${ASSETS_FOLDER})
