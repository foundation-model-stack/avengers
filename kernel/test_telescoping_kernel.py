import torch
import pytest
from telescoping_kernel import IndLinear, IndLinearTransposed


@pytest.mark.parametrize("b", [1, 2, 4])
@pytest.mark.parametrize("l", [4096, 2048])
@pytest.mark.parametrize("n", [8192, 4096])
@pytest.mark.parametrize("h", [16, 32, 64])
@pytest.mark.parametrize("e", [4, 8, 16])
@pytest.mark.parametrize("c", [128, 256])
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_indlinear_forward(b, l, n, h, e, c, d, dtype):
    A = torch.randn(b, l, h, e, d, dtype=dtype, requires_grad=True, device="cuda")
    B = torch.randn(b, n, h, d, dtype=dtype, requires_grad=True, device="cuda")
    M = torch.rand(l, c).mul(n).long().cuda()

    
    Z = (
        B.unsqueeze(-1)
        .expand(-1, -1, -1, -1, c)
        .gather(1, M.view(1, l, 1, 1, c).expand(b, -1, h, d, -1))
    )
    O = A.matmul(Z)
    expected_output = O.clone().detach()

    
    A2 = torch.empty_like(A, requires_grad=True)
    A2.data = A.data
    B2 = torch.empty_like(B, requires_grad=True)
    B2.data = B.data
    indlinear = IndLinear.apply
    O2 = indlinear(A2, B2, M)

    assert torch.allclose(O2, expected_output, atol=1e-2), "Forward pass output mismatch"

@pytest.mark.parametrize("b", [1, 2, 4])
@pytest.mark.parametrize("l", [4096, 2048])
@pytest.mark.parametrize("n", [8192, 4096])
@pytest.mark.parametrize("h", [16, 32, 64])
@pytest.mark.parametrize("e", [4, 8, 16])
@pytest.mark.parametrize("c", [128, 256])
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_indlinear_backward(b, l, n, h, e, c, d, dtype):
    A = torch.randn(b, l, h, e, d, dtype=dtype, requires_grad=True, device="cuda")
    B = torch.randn(b, n, h, d, dtype=dtype, requires_grad=True, device="cuda")
    M = torch.rand(l, c).mul(n).long().cuda()

    # PyTorch computation
    Z = (
        B.unsqueeze(-1)
        .expand(-1, -1, -1, -1, c)
        .gather(1, M.view(1, l, 1, 1, c).expand(b, -1, h, d, -1))
    )
    O = A.matmul(Z)
    loss = O.pow(2).sum()
    loss.backward()

    A_grad_expected = A.grad.clone().detach()
    B_grad_expected = B.grad.clone().detach()


    A2 = torch.empty_like(A, requires_grad=True)
    A2.data = A.data
    B2 = torch.empty_like(B, requires_grad=True)
    B2.data = B.data
    indlinear = IndLinear.apply
    O2 = indlinear(A2, B2, M)
    loss2 = O2.pow(2).sum()
    loss2.backward()

    assert torch.allclose(A2.grad, A_grad_expected, atol=1e-2), "Backward pass A grad mismatch"
    assert torch.allclose(B2.grad, B_grad_expected, atol=1e-2), "Backward pass B grad mismatch"


@pytest.mark.parametrize("b", [1, 2, 4])
@pytest.mark.parametrize("l", [4096, 2048])
@pytest.mark.parametrize("n", [8192, 4096])
@pytest.mark.parametrize("h", [16, 32, 64])
@pytest.mark.parametrize("e", [4, 8, 16])
@pytest.mark.parametrize("c", [128, 256])
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_indlinear_transpose_forward(b, l, n, h, e, c, d, dtype):
    A = torch.randn(b, l, h, e, c,  dtype=dtype, requires_grad=True, device="cuda")
    B = torch.randn(b, n, h, d, dtype=dtype, requires_grad=True, device="cuda")
    M = torch.rand(l, c).mul(n).long().cuda()

    Z = (
        B.unsqueeze(-2)
        .expand(-1, -1, -1, c, -1)
        .gather(1, M.view(1, l, 1, c, 1).expand(b, -1, h, -1, d))
    )
    O = A.matmul(Z)
    expected_output = O.clone().detach()

    
    A2 = torch.empty_like(A, requires_grad=True)
    A2.data = A.data
    B2 = torch.empty_like(B, requires_grad=True)
    B2.data = B.data
    indlinear = IndLinearTransposed.apply
    O2 = indlinear(A2, B2, M)

    assert torch.allclose(O2, expected_output, atol=1e-2), "Forward pass output mismatch"



@pytest.mark.parametrize("b", [1, 2, 4])
@pytest.mark.parametrize("l", [4096, 2048])
@pytest.mark.parametrize("n", [8192, 4096])
@pytest.mark.parametrize("h", [16, 32, 64])
@pytest.mark.parametrize("e", [4, 8, 16])
@pytest.mark.parametrize("c", [128, 256])
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_indlinear_backward(b, l, n, h, e, c, d, dtype):
    A = torch.randn(b, l, h, e, c, dtype=dtype, requires_grad=True, device="cuda")
    B = torch.randn(b, n, h, d, dtype=dtype, requires_grad=True, device="cuda")
    M = torch.rand(l, c).mul(n).long().cuda()

    Z = (
        B.unsqueeze(-2)
        .expand(-1, -1, -1, c, -1)
        .gather(1, M.view(1, l, 1, c, 1).expand(b, -1, h, -1, d))
    )
    O = A.matmul(Z)


    loss = O.pow(2).sum()
    loss.backward()

    A_grad_expected = A.grad.clone().detach()
    B_grad_expected = B.grad.clone().detach()


    A2 = torch.empty_like(A, requires_grad=True)
    A2.data = A.data
    B2 = torch.empty_like(B, requires_grad=True)
    B2.data = B.data
    indlinear = IndLinear.apply
    O2 = indlinear(A2, B2, M)
    loss2 = O2.pow(2).sum()
    loss2.backward()

    assert torch.allclose(A2.grad, A_grad_expected, atol=1e-2), "Backward pass A grad mismatch"
    assert torch.allclose(B2.grad, B_grad_expected, atol=1e-2), "Backward pass B grad mismatch"