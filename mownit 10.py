import jax
import jax.numpy as jnp
import optax
import numpy as np


class PINN:
    def __init__(self, layers):
        self.params = self.init_params(layers)

    def init_params(self, layers):
        params = []
        for i in range(len(layers) - 1):
            w = jax.random.normal(jax.random.PRNGKey(i), (layers[i], layers[i + 1])) * jnp.sqrt(2 / (layers[i] + layers[i + 1]))
            b = jnp.zeros(layers[i + 1])
            params.append((w, b))
        return params

    def forward(self, params, x):
        for w, b in params[:-1]:
            x = jnp.tanh(jnp.dot(x, w) + b)
        w, b = params[-1]
        return jnp.dot(x, w) + b

    def predict(self, x):
        return self.forward(self.params, x)

def residual_loss(params, model, x, omega):
    def single_point_residual(xi):
        u = model.forward(params, xi.reshape(-1, 1))
        u_x = jax.grad(lambda x: model.forward(params, x).sum())(xi.reshape(-1, 1))
        res = u_x - jnp.cos(omega * xi)
        return res**2
    
    residuals = jax.vmap(single_point_residual)(x)
    return jnp.mean(residuals)

def initial_condition_loss(params, model):
    u_0 = model.forward(params, jnp.array([[0.0]]))
    return jnp.mean((u_0 - 0)**2)

def total_loss(params, model, x, omega):
    return residual_loss(params, model, x, omega) + initial_condition_loss(params, model)

def save_results(filename, x, u_true, u_pred, loss_history):
    np.savetxt(f"{filename}_x.csv", x, delimiter=",")
    np.savetxt(f"{filename}_u_true.csv", u_true, delimiter=",")
    np.savetxt(f"{filename}_u_pred.csv", u_pred, delimiter=",")
    np.savetxt(f"{filename}_loss_history.csv", loss_history, delimiter=",")


# Parametry
omega_values = [1, 15]
hidden_layers_architectures = [(2, 16), (4, 64), (5, 128)]
domain = [-2 * np.pi, 2 * np.pi]

for omega in omega_values:
    for layers, neurons in hidden_layers_architectures:
        print(f"Omega: {omega}, Layers: {layers}, Neurons: {neurons}")
        layers = [1] + [neurons] * layers + [1]
        model = PINN(layers)
        optimizer = optax.adam(learning_rate=0.001)
        opt_state = optimizer.init(model.params)

        x_train = jnp.linspace(domain[0], domain[1], 3000 if omega == 15 else 200).reshape(-1, 1)
        epochs = 50000
        loss_history = []

        for epoch in range(epochs):
            loss, grads = jax.value_and_grad(total_loss)(model.params, model, x_train, omega)
            updates, opt_state = optimizer.update(grads, opt_state)
            model.params = optax.apply_updates(model.params, updates)
            if epoch % 10 == 0:
                loss_history.append(loss)

            if epoch % 1000 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

        x_test = np.linspace(domain[0], domain[1], 5000 if omega == 15 else 1000).reshape((-1, 1))
        u_true = (1 / omega) * np.sin(omega * x_test)
        u_pred = model.predict(x_test)

        filename = f"results_omega_{omega}_layers_{len(layers)-2}_neurons_{neurons}"
        save_results(filename, x_test, u_true, u_pred, loss_history)


print("All results have been saved to files.")

