import triton.experimental.tle.language as tle
import torch
import triton
import triton.language as tl

DEVICE_MESH = tle.device_mesh(tle.MeshConfig(device=2))


@triton.jit
def _tle_local_pe_kernel(in_ptr, out_ptr, mesh: tl.constexpr, BLOCK: tl.constexpr):
    local_rank = tle.my_pe(in_ptr)
    tl.static_print("local_rank", local_rank)


class TestLocalPeCount:

    def test_tle_local_pe_kernel(self):
        block = 64
        grid = 2
        N = 64
        with torch.cuda.use_mem_pool(tle.get_mem_pool()):
            x = torch.randn((N, N), dtype=torch.float32, device="cuda")
        y = torch.empty_like(x)

        dis_tensor_ptr = tle.create_comm_tensor(x)

        compiled = _tle_local_pe_kernel.warmup(
            in_ptr=dis_tensor_ptr,
            out_ptr=y,
            mesh=DEVICE_MESH,
            BLOCK=block,
            grid=(grid, ),
            num_ctas=1,
            num_warps=4,
        )
        assert "remote_pointers" in compiled.asm["ttgir"]
        assert "flagcxGetIntraPointerC" in compiled.asm['ptx']

        _tle_local_pe_kernel[(grid, )](in_ptr=dis_tensor_ptr, out_ptr=y, mesh=DEVICE_MESH, BLOCK=block)


TestLocalPeCount().test_tle_local_pe_kernel()
