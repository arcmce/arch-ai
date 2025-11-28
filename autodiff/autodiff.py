import numpy as np


class Tensor:
    def __init__(self, data, requires_grad=False, previous=()):
        self.data = np.array(data, dtype=float)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(data) if self.requires_grad else None
        self.previous = previous

    def backward(self):
        if grad is None:
            if self.data.size != 1:
                raise ValueError()
            grad = np.ones_like(self.data)

        if self.grad is None and self.requires_grad:
            self.grad = np.zeros_like(self.data)
        if self.grad is None:
            self.grad = np.zeros_like(self.data)

        self.grad = self.grad + grad

        # build topo order (post-order DFS)
        topo = []
        visited = set()
        def build(v):
            if id(v) in visited:
                return
            visited.add(id(v))
            for p in v.previous:
                build(p)
            topo.append(v)
        build(self)

        # reverse traversal
        for node in reversed(topo):
            node._backward()

    def __add__(self, other): return add(self, other)
    def __radd__(self, other): return add(other, self)


def _to_tensor(x):
    return x if isinstance(x, Tensor) else Tensor(x, requires_grad=False)

def _unbroadcast(grad, shape):

    if shape == ():
        return grad.sum()

    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)

    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad

def add(a, b):
    a, b = _to_tensor(a), _to_tensor(b)

    # if either parent requires grad, out requires
    out = Tensor(a.data + b.data, requires_grad=a.requires_grad or b.requires_grad, previous=(a, b))

    def _backward():
        g = out.grad
        if a.requires_grad:
            grad_a = _unbroadcast(g, a.data.shape)
            if a.grad is None:
                a.grad = np.zeros_like(a.data)
            a.grad += grad_a
        if b.requires_grad:
            grad_b = _unbroadcast(g, b.data.shape)
            if b.grad is None:
                b.grad = np.zeros_like(b.data)
            b.grad += grad_b

    out._backward = _backward
    return out

if __name__ == "__main__":

    a = Tensor(3.0, requires_grad=True)
    b = Tensor(4.0, requires_grad=True)

    out = a + b

    out.backward(np.ones_like(out.data))

    print(a.grad, b.grad)
