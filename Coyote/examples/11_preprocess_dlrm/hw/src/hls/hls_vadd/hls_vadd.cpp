#include "hls_vadd.hpp"

void hls_vadd(hls::stream<axi_s> &axi_in,
              hls::stream<axi_s> &axi_out)
{
    #pragma HLS INTERFACE ap_ctrl_none port=return
    #pragma HLS INTERFACE axis register port=axi_in  name=s_axi_in
    #pragma HLS INTERFACE axis register port=axi_out name=m_axi_out

    hls::stream<data_t> s_load, q_dense_in, q_sparse_in;
    hls::stream<data_t> q_dense0, q_dense1, q_sparse_out, s_final;
#pragma HLS stream variable=s_load       depth=64
#pragma HLS stream variable=q_dense_in   depth=32
#pragma HLS stream variable=q_sparse_in  depth=32
#pragma HLS stream variable=q_dense0     depth=32
#pragma HLS stream variable=q_dense1     depth=32
#pragma HLS stream variable=q_sparse_out depth=32
#pragma HLS stream variable=s_final      depth=64

    //dataflow 
#pragma HLS dataflow

    LoadData          (axi_in,      s_load);

    Dense_NegsToZero  (s_load,  q_dense0);
    Dense_Log         (q_dense0,    q_dense1);

    Sparse_HexToIntMod(q_dense1, q_sparse_out);

    StoreData         (q_sparse_out,     axi_out);
}

