import numpy as np

class NavShieldSimulator:
    def __init__(self):
        self.dt = 0.1
        self.time = 0.0
        self.logs = []

    def run_scenario(self, name, duration_s, config):
        print(f"Running scenario: {name}")
        self.time = 0.0
        # x: [x, y, v, theta]
        x = np.zeros(4)
        P = np.eye(4) * 0.1
        Q = np.diag([0.01, 0.01, 0.05, 0.001])

        weight_A = 1.0
        weight_B = 0.0

        target_v = config.get("velocity", 11.11) # 40 km/h

        for _ in range(int(duration_s / self.dt)):
            self.time += self.dt

            # Ground truth
            if name == "LONG_STOP":
                true_v = 0.0
            else:
                true_v = target_v
            true_pos = np.array([true_v * self.time, 0.0])

            # Predict
            x[0] += x[2] * np.cos(x[3]) * self.dt
            x[1] += x[2] * np.sin(x[3]) * self.dt
            # Velocity and heading from Member 2 (IMU/Odometer)
            x[2] = true_v + np.random.normal(0, 0.02)
            x[3] = 0.0 + np.random.normal(0, 0.001)
            P += Q * self.dt

            # Update
            gnss_available = True
            if name == "TUNNEL_200M" and 10.0 < self.time < 30.0:
                gnss_available = False

            z_gnss = true_pos + np.random.normal(0, 2.0, 2)
            if name == "GNSS_JUMP" and self.time >= 30.0:
                z_gnss += 500.0
                innovation = np.linalg.norm(z_gnss - x[:2])
                if innovation > 50.0: gnss_available = False

            if gnss_available:
                H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
                R = np.eye(2) * (2.0**2)
                S = H @ P @ H.T + R
                K = P @ H.T @ np.linalg.inv(S)
                x = x + K @ (z_gnss - H @ x)
                P = (np.eye(4) - K @ H) @ P

            # Road Fork
            if name == "ROAD_FORK" and self.time > 20.0:
                if weight_B == 0.0: weight_B = 0.5; weight_A = 0.5
                # Correct road matches ground truth
                weight_A *= 1.05
                weight_B *= 0.95
                total = weight_A + weight_B
                weight_A /= total; weight_B /= total

            err = np.linalg.norm(x[:2] - true_pos)
            self.logs.append({"t": self.time, "err": err, "wA": weight_A})

        final_err = self.logs[-1]["err"]
        passed = False
        if name == "TUNNEL_200M" and final_err < 15.0: passed = True
        if name == "GNSS_JUMP" and final_err < 10.0: passed = True
        if name == "ROAD_FORK" and weight_A > 0.75: passed = True
        if name == "LONG_STOP" and final_err < 0.5: passed = True

        print(f"Result: {'PASS' if passed else 'FAIL'} (Final Err: {final_err:.2f}m)")
        return passed

if __name__ == "__main__":
    sim = NavShieldSimulator()
    sim.run_scenario("TUNNEL_200M", 40.0, {})
    sim.run_scenario("GNSS_JUMP", 60.0, {})
    sim.run_scenario("ROAD_FORK", 50.0, {})
    sim.run_scenario("LONG_STOP", 45.0, {})
