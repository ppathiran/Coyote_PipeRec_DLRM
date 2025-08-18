cmake_minimum_required(VERSION 3.14)
project(dlrm_rdma_multi)

# Just delegate to client and server
add_subdirectory(src/client)
add_subdirectory(src/server)

