cmake_minimum_required (VERSION 3.8)

set(VCPKG_TARGET_TRIPLET x64-windows)
set(VCPKG_CRT_LINKAGE dynamic)
set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_HOME}/scripts/buildsystems/vcpkg.cmake")
project(physx_test VERSION 0.1.0 LANGUAGES C CXX)
add_executable(physx_test main.cpp meshes.h meshes.cpp)

find_package(unofficial-omniverse-physx-sdk CONFIG REQUIRED)
find_package(fastgltf CONFIG REQUIRED)
target_link_libraries(physx_test PRIVATE fastgltf::fastgltf)
target_link_libraries(physx_test PRIVATE unofficial::omniverse-physx-sdk::sdk)

if(TARGET unofficial::omniverse-physx-sdk::gpu-library)
    if(UNIX)
        # Add rpath setting to find so libraries on unix based systems
        set_target_properties(physx_test PROPERTIES
                BUILD_WITH_INSTALL_RPATH TRUE
                INSTALL_RPATH "$ORIGIN"
        )
    endif()
    add_custom_command(TARGET physx_test POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:unofficial::omniverse-physx-sdk::gpu-library>
            $<TARGET_FILE_DIR:physx_test>)
    if(WIN32)
        add_custom_command(TARGET physx_test POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:unofficial::omniverse-physx-sdk::gpu-device-library>
                $<TARGET_FILE_DIR:physx_test>)
    endif()
else()
    message(WARNING "\GPU acceleration library target not defined
 - GPU acceleration will NOT be available!\
")
endif()