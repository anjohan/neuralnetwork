if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.9)
    project(regression)
    enable_language(Fortran)

    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE Release)
    endif()

    find_package(OpenMP)
    if(OpenMP_Fortran_FOUND)
        set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${OpenMP_Fortran_FLAGS}")
    endif()

    if(CMAKE_Fortran_COMPILER_ID MATCHES GNU)
        set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -Wall -Wextra")
        set(CMAKE_Fortran_FLAGS_DEBUG "${CMAKE_Fortran_FLAGS_DEBUG} -fcheck=all")
        set(CMAKE_Fortran_FLAGS_RELEASE "${CMAKE_Fortran_FLAGS_RELEASE} -march=native")
    endif()

else()
    set(CMAKE_Fortran_MODULE_DIRECTORY ${PROJECT_BINARY_DIR})
endif()

file(GLOB sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.*90)

add_library(neuralnetwork ${sources})
