# Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: MIT
if("${CMAKE_MACHINE}" STREQUAL "VersalNet")
    if("${CMAKE_SUBMACHINE}" STREQUAL "VersalNet")
        add_subdirectory(versal_net)
    elseif("${CMAKE_SUBMACHINE}" STREQUAL "Versal_2VE_2VM")
        add_subdirectory(versal_2ve_2vm)
    endif()
endif()
