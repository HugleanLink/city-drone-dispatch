from .model import Network, Order, Route, START
from .sim import Engine


def mechanism_network():
    network=Network((30.21,120.18),(-5,5,-4,4),())
    hub=network.latlon(-1.5,-1.5)
    spoke=network.latlon(1.5,0)
    return Network(network.origin,network.bounds,(Route("H01-S01","H01","S01",hub,spoke,
                                                        network.straight(hub,spoke),0),))


def three_orders(strategy):
    n=Network((30.21,120.18),(-5,5,-4,4),())
    home=n.latlon(0,0)
    p=n.latlon(.5,0)
    a=Order("A",720,720,732,home,n.latlon(2,0),"waimai",1,source="manual")
    b=Order("B",720,720,727,home,p,"waimai",3,source="manual")
    c=Order("C",720,720,734,p,home,"waimai",3,source="manual")
    run=Engine(n,[a,b,c],strategy,{"riders":0},start=720)
    run.cal.tails["rider:demo:01"]=(720,home)
    return run


def battery_pair(disturbed, strategy="lookahead"):
    network=mechanism_network()
    route=network.routes[0]
    run=Engine(network,strategy=strategy,settings={"riders":2,"aircraft":4,"pads":2,"swap":2,"charge":0},start=709)
    for i,rider in enumerate(run.cal.tails):
        run.cal.tails[rider]=(709,route.pickup if i<4 else route.dropoff)
    # Committed flights use the four installed batteries. Four subsequent orders compete for spares.
    fleet=run.hubs["H01"]
    from .sim import Slot
    for plane,battery in fleet["mounted"].items():
        run.cal.slots[plane].append(Slot(plane,709,720,"已承诺航空任务","opening"))
        run.cal.slots[battery].append(Slot(battery,709,1321,"待充","opening"))
    for i in range(4):
        run.add(Order(f"N{i+1}",720,720,743+i,route.pickup,route.dropoff,"waimai",1,source="manual"))
    if disturbed:
        run.limit("battery","H01",3,720,780)
    run.advance(719)
    return run
