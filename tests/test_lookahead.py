from collections import Counter
from dataclasses import replace
from threading import Event
from types import SimpleNamespace

from src.examples import mechanism_network
from src.lookahead import POLICY, WAIT_BUDGET, _window, choose
from src.model import Order
from src.sim import Calendar, Engine, Plan, Slot


class RolloutFixture:
    """Small clock and calendar where dispatch choices have known consequences."""

    _best = staticmethod(Engine._best)
    _ready_rank = staticmethod(Engine._ready_rank)

    def __init__(self, pivot, peers, *, quicker_at=None):
        self.now = pivot.arrival
        self.cal = Calendar()
        self.records = {p.id: {"status": "unready" if p.ready > self.now else "waiting", "order": p}
                        for p in [pivot, *peers]}
        self.routes = {"leg": SimpleNamespace(id="leg")}
        self.lookahead_policy = POLICY
        self.lookahead_stats = Counter()
        self.lookahead_wait_until = {}
        self.lookahead_wait_spent = {}
        self.lookahead_bypasses = {}
        self.lookahead_trace = []
        self.quicker_at = quicker_at

    def _ground(self, order, calendar, now, force):
        rider = "rider:2" if order.id == "peer" and self.quicker_at is not None else "rider:1"
        duration = 5 if order.id == "pivot" and self.quicker_at and now >= self.quicker_at else 10
        start, _ = calendar.next([rider], max(order.ready, now), duration)
        if force and start != now:
            return None
        return Plan("ground", start, start+duration, start+1,
                    (Slot(rider, start, start+duration, "地面直送", order.id),), ())

    def _air_route(self, order, route, calendar, now, force, cutoff, cancelled):
        start, _ = calendar.next(["rider:2"], max(order.ready, now), 5)
        eta = start + 13
        if (force and start != now) or (cutoff is not None and eta > cutoff):
            return None
        return Plan("air", start, eta, start+1, (Slot("rider:2", start, start+5, "首段接驳", order.id),),
                    (), route.id)

    def opts(self, order, calendar, now, *, fastest, cancelled):
        ground = self._ground(order, calendar, now, False)
        air = None if order.ground_only else self._air_route(order, self.routes["leg"], calendar, now, False,
                                                            None, cancelled)
        return ground, air


def order(name, *, arrival=100, ready=100, route=None):
    return Order(name, arrival, ready, None, (0., 0.), (0., .01), "waimai", 1.,
                 route_id=route, candidate_route_ids=(route,) if route else ())


def test_rollout_can_sacrifice_focal_eta_to_free_the_ground_rider():
    pivot, peer = order("pivot", route="leg"), order("peer")
    run = RolloutFixture(pivot, [peer])
    reference = run._ground(pivot, run.cal, run.now, True)
    choice = choose(run, pivot, reference, lambda: False)
    assert choice.plan.mode == "air" and choice.plan.eta > reference.eta
    record = run.lookahead_trace[0]
    assert record["dynamic_projected_eta"][peer.id] == 120
    assert record["selected_projected_eta"][peer.id] == 110
    assert record["selected"]["total_minutes"] == 23
    assert record["candidates"][0]["total_minutes"] == 30
    assert not run.cal.slots
    run.cal.reserve(choice.plan)
    assert run._ground(peer, run.cal, run.now, True).first == run.now


def test_wait_has_a_known_event_and_stops_when_budget_is_spent():
    pivot, peer = order("pivot", route="leg"), order("peer", ready=102)
    run = RolloutFixture(pivot, [peer], quicker_at=102)
    reference = run._ground(pivot, run.cal, run.now, True)
    choice = choose(run, pivot, reference, lambda: False)
    assert choice.kind == "event_wait" and choice.until == peer.ready and choice.plan is None
    assert not run.cal.slots
    run.lookahead_wait_spent[pivot.id] = WAIT_BUDGET
    assert choose(run, pivot, reference, lambda: False).kind == "dynamic"


def test_window_does_not_read_future_arrivals_even_if_marked_unready():
    pivot = order("pivot")
    future = order("future", arrival=101, ready=105)
    known = order("known", ready=102)
    run = RolloutFixture(pivot, [future, known])
    assert _window(run, pivot, POLICY) == [known]
    assert _window(run, pivot, replace(POLICY, diversify=False)) == [known]


def test_projection_respects_an_existing_hold():
    pivot, peer = order("pivot", route="leg"), order("peer")
    run = RolloutFixture(pivot, [peer])
    run.lookahead_wait_until[peer.id] = 120
    reference = run._ground(pivot, run.cal, run.now, True)
    choice = choose(run, pivot, reference, lambda: False)
    # The held peer cannot benefit from freeing a rider before the hold ends.
    assert choice.kind == "dynamic"


def test_ground_wait_constraint_can_reject_a_locally_faster_later_departure():
    pivot, peer = order("pivot"), order("peer", ready=102)
    run = RolloutFixture(pivot, [peer], quicker_at=102)
    plan = run._ground(pivot, run.cal, run.now, True)
    assert choose(run, pivot, plan, lambda: False).kind == "dynamic"
    assert run.lookahead_stats["blocked_ground_wait"] == 1


def test_cancellation_after_rollout_does_not_commit(monkeypatch):
    net = mechanism_network()
    route = net.routes[0]
    pivot = replace(order("pivot"), pickup=route.pickup, dropoff=route.dropoff, kg=3)
    run = Engine(net, [pivot], "lookahead")
    flag = Event()
    from src import sim

    def stop_after_choice(engine, order, plan, cancelled):
        action = choose(engine, order, plan, cancelled)
        flag.set()
        return action

    monkeypatch.setattr(sim, "choose", stop_after_choice)
    run.advance(pivot.arrival, flag.is_set)
    assert run.records[pivot.id]["status"] == "waiting"
    assert not run.cal.slots
    monkeypatch.setattr(sim, "choose", choose)
    flag.clear()
    run.finish(flag.is_set)
    assert run.records[pivot.id]["status"] == "delivered"
    assert sum(e["type"] == "dispatch" for e in run.log) == 1
