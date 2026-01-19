import numpy as np

def curvilinear_abscissa(pos, centerline):
    P = np.array(pos)

    best_s = 0.0
    best_dist = float("inf")
    s_acc = 0.0

    for i in range(len(centerline) - 1):
        A = np.array(centerline[i])
        B = np.array(centerline[i + 1])

        AB = B - A
        AP = P - A

        t = np.dot(AP, AB) / (np.dot(AB, AB) + 1e-9)
        t = np.clip(t, 0.0, 1.0)

        proj = A + t * AB
        d = np.linalg.norm(P - proj)

        if d < best_dist:
            best_dist = d
            best_s = s_acc + t * np.linalg.norm(AB)

        s_acc += np.linalg.norm(AB)

    return best_s

