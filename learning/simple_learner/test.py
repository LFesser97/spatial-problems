from pymatsim.simulation import Simulation


ll = []
for xx in range(10):
    ll.append(xx * 5)

print(ll)

sim = Simulation('/home/andrew/mandl',
                 '/home/andrew/matsim/matsim-0.10.1/matsim-0.10.1.jar')
