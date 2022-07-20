cmake_minimum_required(VERSION 3.0)

set(PROJECT_NAME "EnlightenGAN")

project(${PROJECT_NAME})

if (WIN32)
    if (DEFINED ENV{DEV_LIB_DIR})
        message("Found DEV_LIB_DIR=$ENV{DEV_LIB_DIR}")
    else()
        message(FATAL_ERROR "No Found DEV Library, Please specify DEV Library Path by environment variable DEV_LIB_DIR")
    endif()
    message("WINDOWS")
    include_directories(
        $ENV{DEV_LIB_DIR}/opencv/include
        $ENV{DEV_LIB_DIR}/onnxruntime/include
    )
    
    link_directories(
        $ENV{DEV_LIB_DIR}/opencv/lib
        $ENV{DEV_LIB_DIR}/onnxruntime/lib
    )
    
    link_libraries(
        opencv_world347
        onnxruntime
    )
else()
    message("UNIX")
endif()

file(GLOB_RECURSE SRC *.cpp *.c)
file(GLOB_RECURSE INC *.h *.hpp)

add_library(${PROJECT_NAME} SHARED ${SRC} ${INC})

install(
    TARGETS ${PROJECT_NAME}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ
    LIBRARY DESTINATION lib
)

install(
    FILES ${INC}
    PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ GROUP_WRITE WORLD_READ
    DESTINATION include
)