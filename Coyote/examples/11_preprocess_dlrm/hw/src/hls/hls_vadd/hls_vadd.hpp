#pragma once
#include "hls_stream.h"
#include "ap_axi_sdata.h"
#include <ap_int.h>
#include <hls_math.h>

#define AXI_DATA_BITS 512
typedef ap_axiu<AXI_DATA_BITS, 0, 0, 0> axi_s;

struct data_t {
    ap_uint<AXI_DATA_BITS>      data;
    ap_uint<AXI_DATA_BITS / 8>  keep;
    bool                        last;
};

typedef union { float float32; uint32_t uint32; } conv;

void hls_vadd(hls::stream<axi_s> &axi_in,
              hls::stream<axi_s> &axi_out);


void LoadData(hls::stream<axi_s> &axi_in,
              hls::stream<data_t> &out_stream)
{
#pragma HLS PIPELINE II=1
#pragma HLS INLINE off
    if (!axi_in.empty()) {
        axi_s in = axi_in.read();
        data_t o; o.data = in.data; o.keep = in.keep; o.last = in.last;
        out_stream.write(o);
    }
}

void StoreData(hls::stream<data_t> &in_stream,
               hls::stream<axi_s>  &axi_out)
{
#pragma HLS PIPELINE II=1
#pragma HLS INLINE off
    if (!in_stream.empty()) {
        data_t i = in_stream.read();
        axi_s  o; o.data = i.data; o.keep = i.keep; o.last = i.last;
        axi_out.write(o);
    }
}


/* ─────────────────────────────────────────────── */
/*  In‑line conditional preprocessing stages       */
/* ─────────────────────────────────────────────── */
void Dense_NegsToZero(hls::stream<data_t> &in,
                      hls::stream<data_t> &out)
{
#pragma HLS PIPELINE II=1
#pragma HLS INLINE off
    static unsigned pkt_id = 0;          // 0=dense, 1/2=sparse
    if (!in.empty()) {
        data_t p = in.read();

        if (pkt_id % 3 == 0) {           // dense packet -> apply clamp‑to‑zero
            ap_uint<512> d = p.data;
            for (int i = 0; i < 16; i++) {
#pragma HLS UNROLL
                int32_t v = d(32*i+31, 32*i);
                d(32*i+31, 32*i) = (v < 0) ? 0 : v;
            }
            p.data = d;
        }
        out.write(p);
        pkt_id++;
    }
}


void Dense_Log(hls::stream<data_t> &in,
               hls::stream<data_t> &out)
{
#pragma HLS PIPELINE II=1
#pragma HLS INLINE off
    static unsigned pkt_id = 0;
    if (!in.empty()) {
        data_t p = in.read();

        if (pkt_id % 3 == 0) {           // dense packet -> log(x+1)
            ap_uint<512> d = p.data;
            for (int i = 0; i < 16; i++) {
#pragma HLS UNROLL
                int32_t v = d(32*i+31, 32*i);
                conv c; c.float32 = hls::logf(v + 1.0f);
                d(32*i+31, 32*i) = c.uint32;
            }
            p.data = d;
        }
        out.write(p);
        pkt_id++;
    }
}



void Sparse_HexToIntMod(hls::stream<data_t> &in,
                        hls::stream<data_t> &out)
{
#pragma HLS PIPELINE II=1
#pragma HLS INLINE off
    static unsigned pkt_id = 0;
    if (!in.empty()) {
        data_t p = in.read();

        if (pkt_id % 3 != 0) {           // sparse packets -> value & 0x3FF
            ap_uint<512> d = p.data;
            for (int i = 0; i < 16; i++) {
#pragma HLS UNROLL
                ap_uint<32> v = d(32*i+31, 32*i);
                d(32*i+31, 32*i) = v & 0x3FF;
            }
            p.data = d;
        }
        out.write(p);
        pkt_id++;
    }
}




