from __future__ import annotations

import sys
import threading

import stats


def test_record_hit_is_thread_safe():
    old_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    threads_n = 50
    increments_per_thread = 200
    try:
        for round_no in range(3):
            category = f"round-{round_no}"
            expected_total = threads_n * increments_per_thread

            def worker():
                for _ in range(increments_per_thread):
                    stats.record_hit(category)

            threads = [threading.Thread(target=worker) for _ in range(threads_n)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            got = stats.hit_count(category)
            assert got == expected_total, (
                f"lost increments under concurrent load in round {round_no}: "
                f"expected {expected_total}, got {got} (race on the shared counter)"
            )
    finally:
        sys.setswitchinterval(old_interval)
