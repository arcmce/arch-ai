import numpy as np


class Tensor:
    def __init__(self, data, requires_grad=False, previous=()):
        self.data = np.array(data, dtype=float)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(data) if self.requires_grad else None
        self.previous = previous
        self._backward = lambda: None

    def backward(self, grad=None):
        print("backward() called; seed grad arg:", grad)
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

        print("topo order (reversed):", [type(n).__name__ + ":" + str(n.data) for n in reversed(topo)])

        # reverse traversal
        for node in reversed(topo):
            node._backward()

    def __repr__(self):
        return f"Tensor({self.data}, req={self.requires_grad})"

    def __add__(self, other): return add(self, other)
    def __radd__(self, other): return add(other, self)
    def __mul__(self, other): return mul(self, other)
    def __rmul__(self, other): return mul(other, self)


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
    out_data = a.data + b.data
    # if either parent requires grad, out requires
    out = Tensor(out_data, requires_grad=(a.requires_grad or b.requires_grad), previous=(a, b))

    def _backward():
        g = out.grad

        print(" add._backward g:", g)

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

def mul(a, b):
    a, b = _to_tensor(a), _to_tensor(b)
    out_data = a.data * b.data
    # if either parent requires grad, out requires
    out = Tensor(out_data, requires_grad=(a.requires_grad or b.requires_grad), previous=(a, b))

    def _backward():
        g = out.grad

        print(" mul._backward g:", g)

        if a.requires_grad:
            grad_a = _unbroadcast(g * b.data, a.data.shape)
            if a.grad is None:
                a.grad = np.zeros_like(a.data)
            a.grad += grad_a
        if b.requires_grad:
            grad_b = _unbroadcast(g * a.data, b.data.shape)
            if b.grad is None:
                b.grad = np.zeros_like(b.data)
            b.grad += grad_b

    out._backward = _backward
    return out



if __name__ == "__main__":

    # ----------
    # scalar + scalar
    print("Mini Test 1")
    a = Tensor(3.0, requires_grad=True)
    b = Tensor(4.0, requires_grad=True)

    out = a + b

    out.backward(np.ones_like(out.data))

    print(a.grad, b.grad)
    print()

    # ----------
    # vector + scalar
    print("Mini Test 2")
    a = Tensor(np.array([1.0,2.0,3.0]), requires_grad=True)
    b = Tensor(10.0, requires_grad=True)

    out = a + b

    out.backward(np.ones_like(out.data))

    print(a.grad, b.grad)
    print()

    # ----------
    # scalar, multi branch
    print("Mini Test 3")
    a = Tensor(2.0, requires_grad=True)
    b = Tensor(3.0, requires_grad=True)

    y1 = a * b
    y2 = a + b
    z = y1 + y2
    L = z * z

    # print forward values
    print("forward values:")
    for name, node in [("a",a),("b",b),("y1",y1),("y2",y2),("z",z),("L",L)]:
        print(name, "=", node.data)

    L.backward()

    print("final grads:")
    print("a.grad:", a.grad)
    print("b.grad:", b.grad)
    print("z.grad:", z.grad)
    print("y1.grad:", y1.grad)
    print("y2.grad:", y2.grad)
    print()

