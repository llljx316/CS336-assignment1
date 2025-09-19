import math

def learning_rate_schedule(t, lr_max, lr_min, Tw, Tc):
    if t<Tw:
        return t/Tw*lr_max
    elif Tw <= t and t <= Tc:
        return lr_min + 0.5*(1+ math.cos((t-Tw)/(Tc-Tw)*math.pi))*(lr_max-lr_min)
    else:
        return lr_min