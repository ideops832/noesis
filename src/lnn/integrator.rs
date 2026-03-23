//! ODE integrators: Euler and RK4.

/// Trait for numerical integrators of ordinary differential equations.
pub trait Integrator {
    /// Performs one integration step.
    ///
    /// - `state`: current state vector
    /// - `dt`: time step
    /// - `derivative`: function computing the derivative at a given state
    fn step(&self, state: &[f32], dt: f32, derivative: &dyn Fn(&[f32]) -> Vec<f32>) -> Vec<f32>;
}

/// Forward Euler integrator: `x(t+dt) = x(t) + dt * f(x, t)`.
pub struct EulerIntegrator;

impl Integrator for EulerIntegrator {
    fn step(&self, state: &[f32], dt: f32, derivative: &dyn Fn(&[f32]) -> Vec<f32>) -> Vec<f32> {
        let d = derivative(state);
        state.iter().zip(&d).map(|(s, ds)| s + dt * ds).collect()
    }
}

/// Classic 4th-order Runge-Kutta integrator.
pub struct Rk4Integrator;

impl Integrator for Rk4Integrator {
    fn step(&self, state: &[f32], dt: f32, derivative: &dyn Fn(&[f32]) -> Vec<f32>) -> Vec<f32> {
        let n = state.len();
        let k1 = derivative(state);

        let mut tmp = vec![0.0f32; n];

        // k2 = f(state + dt/2 * k1)
        for i in 0..n {
            tmp[i] = state[i] + dt * 0.5 * k1[i];
        }
        let k2 = derivative(&tmp);

        // k3 = f(state + dt/2 * k2)
        for i in 0..n {
            tmp[i] = state[i] + dt * 0.5 * k2[i];
        }
        let k3 = derivative(&tmp);

        // k4 = f(state + dt * k3)
        for i in 0..n {
            tmp[i] = state[i] + dt * k3[i];
        }
        let k4 = derivative(&tmp);

        // result = state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        (0..n)
            .map(|i| state[i] + dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rk4_vs_euler() {
        // Harmonic oscillator: dx/dt = v, dv/dt = -x
        // Exact solution: x(t) = cos(t), v(t) = -sin(t) for x(0)=1, v(0)=0
        let derivative = |state: &[f32]| -> Vec<f32> {
            vec![state[1], -state[0]] // dx=v, dv=-x
        };

        let initial = vec![1.0f32, 0.0]; // x=1, v=0
        let dt = 0.1;
        let steps = 100; // t = 10.0

        let euler = EulerIntegrator;
        let rk4 = Rk4Integrator;

        let mut state_euler = initial.clone();
        let mut state_rk4 = initial;

        for _ in 0..steps {
            state_euler = euler.step(&state_euler, dt, &derivative);
            state_rk4 = rk4.step(&state_rk4, dt, &derivative);
        }

        // Exact at t=10: x = cos(10) ≈ -0.8391
        let exact_x = 10.0f32.cos();
        let euler_err = (state_euler[0] - exact_x).abs();
        let rk4_err = (state_rk4[0] - exact_x).abs();

        assert!(
            rk4_err < euler_err,
            "RK4 error ({rk4_err:.6}) should be less than Euler error ({euler_err:.6})"
        );
        assert!(rk4_err < 0.001, "RK4 error too large: {rk4_err:.6}");
    }
}
