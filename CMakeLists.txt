if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.9)
    project(neuralnetwork Fortran)

    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE Release)
    endif(NOT CMAKE_BUILD_TYPE)

    if(CMAKE_Fortran_COMPILER_ID MATCHES GNU)
        set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -Wall -Wextra")
        set(CMAKE_Fortran_FLAGS_DEBUG "${CMAKE_Fortran_FLAGS_DEBUG} -fcheck=all")
        set(CMAKE_Fortran_FLAGS_RELEASE "${CMAKE_Fortran_FLAGS_RELEASE} -march=native")
        if(NOT CMAKE_Fortran_COMPILER MATCHES caf)
            message(WARNING "using -fcoarray=single, run FC=caf cmake for parallelisation")
            set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fcoarray=single")
        endif(NOT CMAKE_Fortran_COMPILER MATCHES caf)
    else(CMAKE_Fortran_COMPILER_ID MATCHES GNU)
        message(FATAL_ERROR "Only gfortran supports co_sum as of writing.")
    endif(CMAKE_Fortran_COMPILER_ID MATCHES GNU)

else(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(CMAKE_Fortran_MODULE_DIRECTORY ${PROJECT_BINARY_DIR})
endif(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)

file(GLOB sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.*90)

add_library(neuralnetwork ${sources})

install(TARGETS neuralnetwork DESTINATION lib)
