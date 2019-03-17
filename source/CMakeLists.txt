cmake_minimum_required (VERSION 2.6)
project (trace)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fext-numeric-literals -fpermissive -march=native -mtune=native -g")

#find_package(CUDA REQUIRED)

#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
add_executable(trace main.cpp)
#cuda_add_executable(fft fft.cu)
#cuda_add_cufft_to_target(trace)
target_include_directories(trace PUBLIC "/usr/local/cuda/targets/x86_64-linux/include")

#link_directories("/usr/local/cuda/targets/x86_64-linux/lib")
find_library(cufft_lib cufft PATHS "/usr/local/cuda/targets/x86_64-linux/lib")
find_library(cudart_lib cudart PATHS "/usr/local/cuda/targets/x86_64-linux/lib")
#message(STATUS "cufft_lib: " ${cufft_lib})
target_link_libraries(trace ${cufft_lib})
target_link_libraries(trace ${cudart_lib})
#target_link_libraries(trace cufft)
#target_link_libraries(trace cudart)


