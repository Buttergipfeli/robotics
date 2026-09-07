import time

from real_config import TELEOP_HZ, make_follower, make_leader


def main():
    leader = make_leader()
    follower = make_follower(with_camera=False, max_relative_target=None)
    leader.connect()
    follower.connect()
    print("Teleoperation running at "
          f"{TELEOP_HZ} Hz. Move the leader arm, Ctrl+C stops.")
    try:
        while True:
            start = time.perf_counter()
            action = leader.get_action()
            follower.send_action(action)
            time.sleep(max(0.0, 1 / TELEOP_HZ - (time.perf_counter() - start)))
    except KeyboardInterrupt:
        print("\nStopping teleoperation.")
    finally:
        follower.disconnect()
        leader.disconnect()


if __name__ == "__main__":
    main()
