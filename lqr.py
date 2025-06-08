import numpy as np

def finite_horizon_lqr(A, B, Q, R, Qf, N, x0):
    # Initialize containers
    P = [None] * (N + 1)
    K = [None] * N

    # Terminal condition
    P[N] = Qf

    # Backward Riccati recursion
    for t in reversed(range(N)):
        Pt1 = P[t+1]
        K[t] = (R + B**2 * Pt1)**(-1) * B * A * Pt1
        P[t] = Q + K[t]**2 * R + (A - B * K[t])**2 * Pt1

    # Forward rollout
    x = [x0]
    u = []
    for t in range(N):
        ut = -K[t] * x[t]
        u.append(ut)
        x_next = A * x[t] + B * ut
        x.append(x_next)

    # Compute cost
    cost = 0
    for t in range(N):
        cost += Q * x[t]**2 + R * u[t]**2
    cost += Qf * x[N]**2

    return {
        'P': P,
        'K': K,
        'x': x,
        'u': u,
        'cost': cost
    }

# === Parameters for Case 1 ===
A = 2
B = 1
Q = 2
R = 20
Qf = 50
N = 4
x0 = 100

result = finite_horizon_lqr(A, B, Q, R, Qf, N, x0)

print("\nP_t (Cost-to-go matrices):")
for t, Pt in enumerate(result['P']):
    print(f"P[{t}] = {Pt:.4f}")

print("\nK_t (Feedback gains):")
for t, Kt in enumerate(result['K']):
    print(f"K[{t}] = {Kt:.4f}")

print("\nOptimal control sequence u_t:")
for t, ut in enumerate(result['u']):
    print(f"u[{t}] = {ut:.4f}")

print("\nResulting state trajectory x_t:")
for t, xt in enumerate(result['x']):
    print(f"x[{t}] = {xt:.4f}")

print(f"\nTotal optimal cost: {result['cost']:.4f}")
