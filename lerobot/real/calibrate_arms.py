import sys

from real_config import make_follower, make_leader


def calibrate_follower():
    print("Calibrating follower...")
    robot = make_follower(with_camera=False, max_relative_target=None)
    robot.connect(calibrate=False)
    robot.calibrate()
    robot.disconnect()
    print("Follower calibrated.")


def calibrate_leader():
    print("Calibrating leader...")
    leader = make_leader()
    leader.connect(calibrate=False)
    leader.calibrate()
    leader.disconnect()
    print("Leader calibrated.")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "both"
    if target not in ("follower", "leader", "both"):
        sys.exit("usage: calibrate_arms.py [follower|leader|both]")
    if target in ("follower", "both"):
        calibrate_follower()
    if target in ("leader", "both"):
        calibrate_leader()
