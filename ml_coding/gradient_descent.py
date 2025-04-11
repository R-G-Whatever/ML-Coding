class GradientDescent:
    def __init__(self, f, grad_f, lr = 0.01, num_iter = 1000, tol = 1e-6):

        self.f = f
        self.grad_f = grad_f
        self.learning_rate = lr
        self.num_iter = num_iter
        self.tol = tol

    def optimize(self, x_init):
        x = x_init
        history = []

        for i in range(self.num_iter):
            grad = self.grad_f(x)
            x_new = x - self.learning_rate * grad

            history.append(self.f(x_new))

            if abs(self.f(x_new) - self.f(x)) < self.tol:
                break
            x = x_new
        return x, history

def square_func(x):
    return x**2

def grad_square_func(x):
    return 2 * x

if __name__ == "__main__":
    x_init = 10.0
    dg = GradientDescent(square_func, grad_square_func, lr = 0.1)
    optimized_x, hist = dg.optimize(x_init)
    print(hist)
