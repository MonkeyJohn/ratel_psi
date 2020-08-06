import yaml
import numpy as np

from carmodel import BrushTireModel, LinearTireModel
from renderer import _Renderer


def simulation():
    stream = open('/home/xingjiansheng/Documents/src/workplace_xjs/drift/my_drift/car.yaml','r')
    params = yaml.load(stream)
    model = BrushTireModel(params, 1.37, 1.96)
    renderer = _Renderer(params)
    # print(params)

    state0 = np.zeros(6)
    action_hardcode = np.array([15, 0.0])
    
    iterations = 1500
    dt = 0.03
    flag = True
    for i in range(iterations):
        print(state0[3])
        if state0[3] > 12 and flag:
            action_hardcode[1] = 0.3
            flag = False
        state0 = model.state_transition(state0, action_hardcode, dt)
        renderer.update(state0,action_hardcode)


if __name__ == "__main__":
    simulation()
